#!/usr/bin/env python3

"""Run one independent V2 5.5 mm D38999 initialization gate process.

The V2 asset already contains the baked continuous collision proxies.  This
runner never creates or replaces collision geometry at runtime.  Before
physics starts it only authors the predeclared collision-group filter between
the solid coupling-nut grasp proxy and the loose-plug body colliders.
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

from d38999_multilayer_init_timeline_probe import (
    BODY_PATH,
    FIXED_PATH,
    NUT_PATH,
    OWNER_PATHS,
    ROOT,
    _aggregate_contacts,
    _contact_inventory,
    _repository,
    _set_initial_translation,
    _sha256,
    _stage_snapshot,
    _view_snapshot,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_init_gate_v2"
TASK_ID = "DYN-A1-EVENT-ONSET-CALIBRATION-V2"
HYPOTHESIS_ID = "A1-V2-H1-COOKED-SURFACE-AND-CONTACT-MARGIN-ONSET"
START_SEPARATION_M = 0.0055
INTERNAL_DATUM_TARGET_M = 1.0e-05
ABNORMAL_ANGULAR_SPEED_RAD_S = 1.0e-06
EXPECTED_ACTIVE_COLLIDER_COUNT = 270
EXPECTED_FIXED_COLLIDER_COUNT = 199
EXPECTED_BODY_COLLIDER_COUNT = 70
EXPECTED_NUT_COLLIDER_COUNT = 1
EXPECTED_BAKED_PLUG_GUIDE_COLLIDER_COUNT = 64
GRIP_PATH = NUT_PATH + "/CouplingNutGraspCollision"
VISIBLE_PLUG_GUIDE_PATH = BODY_PATH + "/MatingShell/ContinuousPlugGuide"
BAKED_PLUG_GUIDE_ROOT = BODY_PATH + "/MatingShell/ContinuousPlugGuideCollision"

MASTER_RELATIVE = Path("src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
PHYSICAL_RELATIVE = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
ACCEPTANCE_RELATIVE = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
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
FINE_OFFSET_RESULT_RELATIVE = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/FINE_OFFSET_FIX_RESULT.json"
)
OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2"
)
EXPECTED_SHA256 = {
    "master_contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
    "acceptance_contract": "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    "authorized_overrides": "392766e8eceb85a3c910b118c2ad998aef891a74e58c31cd94e383c9908535ce",
    "model": "d5bcc5e8b28e31912f65cd87a0bbe5d7a035744f7f7d8c7b785e17cdad382a6e",
    "mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
    "fine_offset_result": "baee361e50a3a8db8109b0a809ae88a32de4907f35e9e96652c2e00d9b186dd6",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--physical-contract", required=True)
    parser.add_argument("--acceptance-contract", required=True)
    parser.add_argument("--authorized-overrides", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--fine-offset-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


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
    expected_output = (
        repository
        / OUTPUT_ROOT_RELATIVE
        / f"INIT_GATE_RUN_{arguments.run_index:02d}"
    ).resolve()
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
    v2 = state.get("autonomous_dynamic_closeout_v2", {})
    fine_fix = v2.get("fine_collision_offset_fix", {})
    initialization = v2.get("initialization_gate", {})
    required_state = {
        "task_id": (state.get("task_id"), "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2"),
        "status": (state.get("status"), "VALIDATING"),
        "phase": (state.get("phase"), "DYN_A1_EVENT_ONSET_CALIBRATION_V2"),
        "current_node": (v2.get("current_node"), TASK_ID),
        "fine_fix_status": (
            fine_fix.get("status"),
            "STATIC_PASS_AWAITING_DYNAMIC_VALIDATION",
        ),
        "fine_fix_model": (
            fine_fix.get("assembly_control_sha256_after"),
            EXPECTED_SHA256["model"],
        ),
        "queue_status": (queue.get("status"), "VALIDATING"),
        "queue_task": (queue.get("current_task"), TASK_ID),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required_state.items()
        if actual != expected
    }
    completed = int(initialization.get("processes_completed", 0))
    started = int(initialization.get("processes_started", 0))
    if completed != arguments.run_index - 1:
        mismatches["initialization.processes_completed"] = {
            "actual": completed,
            "expected": arguments.run_index - 1,
        }
    if started not in (arguments.run_index - 1, arguments.run_index):
        mismatches["initialization.processes_started"] = {
            "actual": started,
            "expected_one_of": [arguments.run_index - 1, arguments.run_index],
        }
    if mismatches:
        raise PermissionError(f"V2 initialization state guard failed: {mismatches}")

    paths = {
        "master_contract": _frozen_path(
            arguments.contract,
            MASTER_RELATIVE,
            EXPECTED_SHA256["master_contract"],
            "master contract",
        ),
        "physical_contract": _frozen_path(
            arguments.physical_contract,
            PHYSICAL_RELATIVE,
            EXPECTED_SHA256["physical_contract"],
            "physical contract",
        ),
        "acceptance_contract": _frozen_path(
            arguments.acceptance_contract,
            ACCEPTANCE_RELATIVE,
            EXPECTED_SHA256["acceptance_contract"],
            "acceptance contract",
        ),
        "authorized_overrides": _frozen_path(
            arguments.authorized_overrides,
            OVERRIDES_RELATIVE,
            EXPECTED_SHA256["authorized_overrides"],
            "authorized overrides",
        ),
        "model": _frozen_path(
            arguments.model,
            MODEL_RELATIVE,
            EXPECTED_SHA256["model"],
            "assembly-control model",
        ),
        "mapping": _frozen_path(
            arguments.mapping,
            MAPPING_RELATIVE,
            EXPECTED_SHA256["mapping"],
            "model mapping",
        ),
        "fine_offset_result": _frozen_path(
            arguments.fine_offset_result,
            FINE_OFFSET_RESULT_RELATIVE,
            EXPECTED_SHA256["fine_offset_result"],
            "fine-offset result",
        ),
    }
    master = yaml.safe_load(paths["master_contract"].read_text(encoding="utf-8"))
    physical = yaml.safe_load(paths["physical_contract"].read_text(encoding="utf-8"))
    acceptance = yaml.safe_load(paths["acceptance_contract"].read_text(encoding="utf-8"))
    overrides = yaml.safe_load(paths["authorized_overrides"].read_text(encoding="utf-8"))
    result = json.loads(paths["fine_offset_result"].read_text(encoding="utf-8"))
    if master["authoritative_inputs"]["physical_model_contract"]["sha256"] != EXPECTED_SHA256["physical_contract"]:
        raise PermissionError("master-to-physical contract lineage changed")
    if master["authoritative_inputs"]["physical_acceptance_contract"]["sha256"] != EXPECTED_SHA256["acceptance_contract"]:
        raise PermissionError("master-to-acceptance contract lineage changed")
    if result.get("classification") != "EXPLICIT_FINE_COLLISION_OFFSETS_AUTHORED":
        raise PermissionError("fine-offset result classification changed")
    if result.get("assembly_control", {}).get("sha256_after") != EXPECTED_SHA256["model"]:
        raise PermissionError("fine-offset result model lineage changed")
    if overrides.get("initial_nut_grasp_collision_enabled") is not True:
        raise PermissionError("authorized nut grasp collider is no longer enabled")

    formal_limits = master["acceptance_limits"]
    shared = acceptance["shared_numeric_profile"]
    p1 = acceptance["benches"]["P1"]
    thresholds = {
        "datum_tolerance_m": float(p1["pass"]["nominal_event_position_tolerance_m"]),
        "fixed_drift_limit_m": float(
            formal_limits["maximum_fixed_receptacle_translation_drift_m"]
        ),
        "hard_penetration_limit_m": float(
            formal_limits["maximum_noncompliant_hard_penetration_m"]
        ),
        "contact_offset_m": float(shared["fine_contact_offset_m"]),
        "rest_offset_m": float(shared["rest_offset_m"]),
        "physics_rate_hz": int(shared["physics_rate_hz"]),
        "gravity_m_s2": float(
            p1["inputs"]["component_driver_profile"]["gravity_magnitude_m_s2"]
        ),
    }
    expected_thresholds = {
        "datum_tolerance_m": 5.0e-05,
        "fixed_drift_limit_m": 5.0e-06,
        "hard_penetration_limit_m": 5.0e-05,
        "contact_offset_m": 1.0e-05,
        "rest_offset_m": 0.0,
        "physics_rate_hz": 240,
        "gravity_m_s2": 0.0,
    }
    if thresholds != expected_thresholds:
        raise PermissionError(
            f"frozen initialization thresholds changed: {thresholds} != {expected_thresholds}"
        )
    if physical["solver_profile"]["fine_connector_contact_offset_m"] != thresholds["contact_offset_m"]:
        raise PermissionError("physical and acceptance contact offsets differ")
    if physical["solver_profile"]["rest_offset_m"] != thresholds["rest_offset_m"]:
        raise PermissionError("physical and acceptance rest offsets differ")
    return {
        "output": output,
        "paths": paths,
        "master": master,
        "physical": physical,
        "acceptance": acceptance,
        "thresholds": thresholds,
    }


def _collision_inventory(stage: Any, usd_physics: Any, physx_schema: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    prims = [prim for prim in stage.Traverse() if prim.HasAPI(usd_physics.CollisionAPI)]
    rows: list[dict[str, Any]] = []
    for prim in prims:
        path = str(prim.GetPath())
        contact = prim.GetAttribute("physxCollision:contactOffset")
        rest = prim.GetAttribute("physxCollision:restOffset")
        role = prim.GetAttribute("kcg:collisionRole")
        rows.append(
            {
                "path": path,
                "owner": (
                    "fixed_receptacle"
                    if path.startswith(FIXED_PATH + "/")
                    else "body_assembly"
                    if path.startswith(BODY_PATH + "/")
                    else "coupling_nut"
                    if path.startswith(NUT_PATH + "/")
                    else "UNEXPECTED"
                ),
                "collision_role": str(role.Get()) if role and role.Get() is not None else "UNLABELED",
                "physx_collision_api_applied": bool(
                    prim.HasAPI(physx_schema.PhysxCollisionAPI)
                ),
                "contact_offset_authored": bool(
                    contact and contact.HasAuthoredValueOpinion()
                ),
                "contact_offset_m": (
                    float(contact.Get())
                    if contact and contact.Get() is not None
                    else None
                ),
                "rest_offset_authored": bool(rest and rest.HasAuthoredValueOpinion()),
                "rest_offset_m": (
                    float(rest.Get()) if rest and rest.Get() is not None else None
                ),
            }
        )
    rows.sort(key=lambda row: row["path"])
    return prims, rows


def _validate_collision_inventory(
    stage: Any,
    prims: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    usd_physics: Any,
) -> dict[str, Any]:
    owner_counts = {
        owner: sum(row["owner"] == owner for row in rows)
        for owner in ("fixed_receptacle", "body_assembly", "coupling_nut", "UNEXPECTED")
    }
    expected_owner_counts = {
        "fixed_receptacle": EXPECTED_FIXED_COLLIDER_COUNT,
        "body_assembly": EXPECTED_BODY_COLLIDER_COUNT,
        "coupling_nut": EXPECTED_NUT_COLLIDER_COUNT,
        "UNEXPECTED": 0,
    }
    offset_failures = [
        dict(row)
        for row in rows
        if not row["physx_collision_api_applied"]
        or not row["contact_offset_authored"]
        or not row["rest_offset_authored"]
        or not math.isclose(
            float(row["contact_offset_m"]),
            float(thresholds["contact_offset_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(row["rest_offset_m"]),
            float(thresholds["rest_offset_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    visible_guide = stage.GetPrimAtPath(VISIBLE_PLUG_GUIDE_PATH)
    visible_guide_collision_enabled = bool(
        visible_guide and visible_guide.HasAPI(usd_physics.CollisionAPI)
    )
    baked_guide_count = sum(
        str(prim.GetPath()).startswith(BAKED_PLUG_GUIDE_ROOT + "/")
        for prim in prims
    )
    passed = bool(
        len(prims) == EXPECTED_ACTIVE_COLLIDER_COUNT
        and owner_counts == expected_owner_counts
        and not offset_failures
        and not visible_guide_collision_enabled
        and baked_guide_count == EXPECTED_BAKED_PLUG_GUIDE_COLLIDER_COUNT
    )
    return {
        "active_collider_count": len(prims),
        "expected_active_collider_count": EXPECTED_ACTIVE_COLLIDER_COUNT,
        "owner_counts": owner_counts,
        "expected_owner_counts": expected_owner_counts,
        "offset_failure_count": len(offset_failures),
        "offset_failures": offset_failures,
        "visible_plug_guide_has_collision_api": visible_guide_collision_enabled,
        "baked_plug_guide_collider_count": baked_guide_count,
        "runtime_geometry_created_count": 0,
        "old_runtime_proxy_builder_called": False,
        "pass": passed,
    }


def _author_nut_body_filter(stage: Any, rows: Sequence[Mapping[str, Any]], usd_geom: Any, usd_physics: Any, sdf: Any) -> dict[str, Any]:
    body_paths = sorted(
        str(row["path"]) for row in rows if row["owner"] == "body_assembly"
    )
    nut_paths = sorted(
        str(row["path"]) for row in rows if row["owner"] == "coupling_nut"
    )
    if len(body_paths) != EXPECTED_BODY_COLLIDER_COUNT or nut_paths != [GRIP_PATH]:
        raise RuntimeError(
            f"unexpected filter members: body={len(body_paths)}, nut={nut_paths}"
        )
    root = "/World/D38999V2InitializationCollisionGroups"
    usd_geom.Scope.Define(stage, root)

    def define(name: str, members: Sequence[str]) -> tuple[str, Any]:
        path = root + "/" + name
        group = usd_physics.CollisionGroup.Define(stage, path)
        group.CreateInvertFilteredGroupsAttr(False)
        collection = group.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr("explicitOnly")
        collection.CreateIncludeRootAttr(False)
        collection.CreateIncludesRel().SetTargets([sdf.Path(value) for value in members])
        return path, group

    body_group_path, body_group = define("NonGripLoosePlug", body_paths)
    nut_group_path, nut_group = define("AuthorizedCouplingNutGrip", nut_paths)
    nut_group.CreateFilteredGroupsRel().AddTarget(sdf.Path(body_group_path))
    authored_body = sorted(
        str(path)
        for path in body_group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
    )
    authored_nut = sorted(
        str(path)
        for path in nut_group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
    )
    authored_targets = sorted(
        str(path) for path in nut_group.GetFilteredGroupsRel().GetTargets()
    )
    if authored_body != body_paths or authored_nut != nut_paths or authored_targets != [body_group_path]:
        raise RuntimeError("V2 initialization nut/body collision filter readback failed")
    return {
        "enabled": True,
        "authored_before_physics": True,
        "controller_input": False,
        "mode": "explicit_pair_filter_no_contact_truth",
        "body_group_path": body_group_path,
        "nut_group_path": nut_group_path,
        "body_collider_count": len(authored_body),
        "nut_collider_count": len(authored_nut),
        "filtered_group_targets": authored_targets,
        "authorized_grip_collider": GRIP_PATH,
    }


def _pair_class(row: Mapping[str, Any]) -> str:
    paths = [str(value) for value in row["collider_paths"]]
    has_fixed = any(path.startswith(FIXED_PATH + "/") for path in paths)
    has_body = any(path.startswith(BODY_PATH + "/") for path in paths)
    has_nut = any(path.startswith(NUT_PATH + "/") for path in paths)
    if has_fixed and (has_body or has_nut):
        return "fixed_receptacle_vs_loose_plug"
    if has_body and has_nut:
        return "loose_plug_body_vs_coupling_nut"
    return "other_connector_contact"


def _state_metrics(timeline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nonfinite_count = 0
    abnormal_angular_count = 0
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0
    for stage_id in ("T4", "T5", "T7"):
        for body in timeline[stage_id]["bodies"].values():
            for key in ("position_m", "orientation_wxyz", "linear_velocity_m_s", "angular_velocity_rad_s"):
                values = np.asarray(body[key], dtype=np.float64)
                nonfinite_count += int(np.size(values) - np.count_nonzero(np.isfinite(values)))
            linear = float(np.linalg.norm(body["linear_velocity_m_s"]))
            angular = float(np.linalg.norm(body["angular_velocity_rad_s"]))
            maximum_linear_speed = max(maximum_linear_speed, linear)
            maximum_angular_speed = max(maximum_angular_speed, angular)
            abnormal_angular_count += int(angular > ABNORMAL_ANGULAR_SPEED_RAD_S)
    return {
        "nonfinite_scalar_count": nonfinite_count,
        "abnormal_angular_velocity_sample_count": abnormal_angular_count,
        "abnormal_angular_speed_threshold_rad_s": ABNORMAL_ANGULAR_SPEED_RAD_S,
        "maximum_linear_speed_m_s": maximum_linear_speed,
        "maximum_angular_speed_rad_s": maximum_angular_speed,
        "pass": nonfinite_count == 0 and abnormal_angular_count == 0,
    }


def _run(arguments: argparse.Namespace, frozen: Mapping[str, Any], application: Any) -> dict[str, Any]:
    import carb.logging
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    World.clear_instance()
    context = omni.usd.get_context()
    if context.get_stage() is not None:
        context.close_stage()
        application.update()
    if context.open_stage(str(frozen["paths"]["model"])) is not True:
        raise RuntimeError("failed to open V2 assembly-control asset")
    for _ in range(3):
        application.update()
    thresholds = frozen["thresholds"]
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / float(thresholds["physics_rate_hz"]),
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    collision_prims, collision_rows = _collision_inventory(stage, UsdPhysics, PhysxSchema)
    collision_validation = _validate_collision_inventory(
        stage, collision_prims, collision_rows, thresholds, UsdPhysics
    )
    if not collision_validation["pass"]:
        raise RuntimeError(f"V2 collision inventory failed: {collision_validation}")
    collision_filter = _author_nut_body_filter(
        stage, collision_rows, UsdGeom, UsdPhysics, Sdf
    )
    _set_initial_translation(stage, BODY_PATH, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, UsdGeom, Gf)
    timeline: dict[str, dict[str, Any]] = {}
    timeline["T3"] = _stage_snapshot(
        "T3",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="V2 baked asset immediately before World.reset",
    )
    for owner in OWNER_PATHS.values():
        prim = stage.GetPrimAtPath(owner)
        if not prim:
            raise RuntimeError(f"missing contact-report owner {owner}")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    world.get_physics_context().set_gravity(float(thresholds["gravity_m_s2"]))

    log_rows: list[dict[str, Any]] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line_number: int, message: str) -> None:
        text = str(message)
        lowered = text.lower()
        if (
            ("physicsusd" in lowered and ("error" in lowered or "failed" in lowered))
            or ("solver" in lowered and ("error" in lowered or "failed" in lowered))
            or "adjusted the thickness" in lowered
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
        os.write(2, f"A1_V2_INIT_HEARTBEAT run={arguments.run_index} stage=pre_reset\n".encode())
        world.reset()
        interface = get_physx_simulation_interface()
        timeline["T4"] = _stage_snapshot(
            "T4",
            stage,
            UsdGeom,
            UsdPhysics,
            explicit_step_count=0,
            semantic="immediately after World.reset before RigidPrim initialization",
        )
        contacts_t4 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T4")
        views = {
            label: RigidPrim(
                prim_paths_expr=path,
                name=f"v2_init_{arguments.run_index}_{label}",
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
            semantic="after RigidPrim initialization",
        )
        contacts_t5 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T5")
        world.step(render=False)
        timeline["T7"] = _view_snapshot(
            "T7",
            views,
            explicit_step_count=1,
            semantic="after first explicit physics step",
        )
        contacts_t7 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T7")
        os.write(2, f"A1_V2_INIT_HEARTBEAT run={arguments.run_index} stage=first_step_complete\n".encode())
        world.stop()
    finally:
        logging.remove_logger(logger_handle)

    contacts_by_stage = {"T4": contacts_t4, "T5": contacts_t5, "T7": contacts_t7}
    for rows in contacts_by_stage.values():
        for row in rows:
            row["evaluation_pair_class"] = _pair_class(row)
    aggregate = _aggregate_contacts(tuple(contacts_by_stage.values()))
    interpart_counts = {
        stage_id: sum(
            row["evaluation_pair_class"] == "fixed_receptacle_vs_loose_plug"
            for row in rows
        )
        for stage_id, rows in contacts_by_stage.items()
    }
    internal_counts = {
        stage_id: sum(
            row["evaluation_pair_class"] == "loose_plug_body_vs_coupling_nut"
            for row in rows
        )
        for stage_id, rows in contacts_by_stage.items()
    }
    datum_errors = {
        stage_id: abs(float(timeline[stage_id]["datum_error_m"]))
        for stage_id in ("T3", "T4", "T5", "T7")
    }
    fixed_reference = np.asarray(
        timeline["T3"]["bodies"]["fixed_receptacle"]["position_m"], dtype=np.float64
    )
    fixed_drift = {
        stage_id: float(
            np.linalg.norm(
                np.asarray(
                    timeline[stage_id]["bodies"]["fixed_receptacle"]["position_m"],
                    dtype=np.float64,
                )
                - fixed_reference
            )
        )
        for stage_id in ("T4", "T5", "T7")
    }
    maximum_penetration_m = max(
        (
            max(0.0, -float(row["minimum_separation_m"]))
            for rows in contacts_by_stage.values()
            for row in rows
            if row["minimum_separation_m"] is not None
        ),
        default=0.0,
    )
    physicsusd_errors = [
        row for row in log_rows if "physicsusd" in row["message"].lower()
    ]
    solver_errors = [
        row for row in log_rows if "solver" in row["message"].lower()
    ]
    thickness_warnings = [
        row for row in log_rows if "adjusted the thickness" in row["message"].lower()
    ]
    state_metrics = _state_metrics(timeline)
    handles_valid = all(
        bool(timeline["T5"]["bodies"][label]["physics_handle_valid"])
        for label in OWNER_PATHS
    )
    gates = {
        "datum_pass": max(datum_errors.values()) <= float(thresholds["datum_tolerance_m"]),
        "interpart_contact_zero_pass": max(interpart_counts.values()) == 0,
        "internal_nut_body_contact_zero_pass": max(internal_counts.values()) == 0,
        "fixed_drift_pass": max(fixed_drift.values()) <= float(thresholds["fixed_drift_limit_m"]),
        "hard_penetration_pass": maximum_penetration_m <= float(thresholds["hard_penetration_limit_m"]),
        "finite_and_angular_velocity_pass": bool(state_metrics["pass"]),
        "physics_handles_pass": handles_valid,
        "collision_inventory_and_offsets_pass": bool(collision_validation["pass"]),
        "physicsusd_error_pass": len(physicsusd_errors) == 0,
        "solver_error_pass": len(solver_errors) == 0,
        "thickness_warning_pass": len(thickness_warnings) == 0,
        "object_pose_write_pass": True,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "run_index": arguments.run_index,
        "status": "PASS" if passed else "FAIL",
        "classification": (
            "V2_INITIALIZATION_GATE_INDIVIDUAL_PASS"
            if passed
            else "V2_INITIALIZATION_GATE_INDIVIDUAL_FAIL"
        ),
        "individual_initialization_passed": passed,
        "timeline": timeline,
        "datum_error_by_stage_m": datum_errors,
        "maximum_abs_datum_error_m": max(datum_errors.values()),
        "internal_datum_target_m": INTERNAL_DATUM_TARGET_M,
        "fixed_receptacle_drift_by_stage_m": fixed_drift,
        "maximum_fixed_receptacle_translation_drift_m": max(fixed_drift.values()),
        "interpart_connector_contact_pair_count_by_stage": interpart_counts,
        "internal_nut_body_contact_pair_count_by_stage": internal_counts,
        "maximum_noncompliant_hard_penetration_m": maximum_penetration_m,
        "state_metrics": state_metrics,
        "collision_inventory_validation": collision_validation,
        "collision_offset_inventory": collision_rows,
        "nut_body_collision_filter": collision_filter,
        "contacts_by_stage": contacts_by_stage,
        "contact_statistics": aggregate,
        "runtime_log_evidence": {
            "retained_messages": log_rows,
            "physicsusd_error_count": len(physicsusd_errors),
            "solver_error_count": len(solver_errors),
            "adjusted_thickness_warning_count": len(thickness_warnings),
        },
        "gates": gates,
        "thresholds": dict(thresholds),
        "initial_pose_writes_before_physics_start_count": 2,
        "object_pose_write_after_physics_start_count": 0,
        "explicit_physics_step_count": 1,
        "isaac_process_count": 1,
        "simulation_started": True,
        "source_asset_written": False,
        "runtime_geometry_created_count": 0,
        "old_runtime_proxy_builder_called": False,
        "controller_consumed_object_truth": False,
        "controller_consumed_contact_names": False,
        "controller_consumed_contact_normals": False,
        "controller_consumed_event_truth": False,
        "posthoc_contact_truth_for_scoring_only": True,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
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
                    "run_index": arguments.run_index,
                    "status": "PREFLIGHT_PASS",
                    "input_sha256": EXPECTED_SHA256,
                    "thresholds": frozen["thresholds"],
                    "output": str(frozen["output"]),
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
        portable = Path(
            tempfile.mkdtemp(
                prefix=f"kcg-a1-v2-init-{arguments.run_index:02d}-", dir="/tmp"
            )
        )
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
        report = _run(arguments, frozen, application)
        report["input_sha256"] = dict(EXPECTED_SHA256)
        report["post_run_sha256"] = {
            key: _sha256(path) for key, path in frozen["paths"].items()
        }
        report["frozen_inputs_unchanged"] = (
            report["post_run_sha256"] == EXPECTED_SHA256
        )
        report["kit_portable_root"] = str(portable)
        report["runner_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen input SHA-256 changed during initialization gate")
        exit_code = 0 if report["individual_initialization_passed"] else 3
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "run_index": arguments.run_index,
            "status": "ERROR",
            "classification": "V2_INITIALIZATION_GATE_RUNTIME_ERROR",
            "individual_initialization_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "object_pose_write_after_physics_start_count": 0,
            "source_asset_written": False,
            "dynamic_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(
        json.dumps(
            {
                "task_id": report.get("task_id"),
                "run_index": report.get("run_index"),
                "status": report.get("status"),
                "classification": report.get("classification"),
                "individual_initialization_passed": report.get(
                    "individual_initialization_passed"
                ),
                "maximum_abs_datum_error_m": report.get(
                    "maximum_abs_datum_error_m"
                ),
                "maximum_fixed_receptacle_translation_drift_m": report.get(
                    "maximum_fixed_receptacle_translation_drift_m"
                ),
                "interpart_connector_contact_pair_count_by_stage": report.get(
                    "interpart_connector_contact_pair_count_by_stage"
                ),
                "runtime_log_evidence": report.get("runtime_log_evidence"),
                "error": report.get("error"),
            },
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
