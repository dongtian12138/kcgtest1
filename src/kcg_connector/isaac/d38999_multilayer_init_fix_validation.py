#!/usr/bin/env python3

"""Run one independent post-fix DYN-A1 initialization validation process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Sequence

import yaml

from d38999_multilayer_init_timeline_probe import (
    BODY_PATH,
    EXPECTED_SHA256,
    FIXED_PATH,
    NUT_PATH,
    OWNER_PATHS,
    START_SEPARATION_M,
    TOLERANCE_M,
    _aggregate_contacts,
    _contact_inventory,
    _finite,
    _frozen_path,
    _repository,
    _set_initial_translation,
    _sha256,
    _stage_snapshot,
    _view_snapshot,
    CONTRACT_RELATIVE,
    MAPPING_RELATIVE,
    MODEL_RELATIVE,
)
from d38999_multilayer_runtime_collision import (
    configure_continuous_plug_guide_runtime_collision,
)


SCHEMA_VERSION = "kcg_d38999_multilayer_init_fix_validation_v1"
TASK_ID = "DYN-A1-INIT-ROOTCAUSE"
OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/agent_control/tasks/DYN-A1-INIT-ROOTCAUSE"
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    return parser.parse_args(argv)


def _authorize(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = _repository()
    expected_output = (
        repository / OUTPUT_ROOT_RELATIVE / f"VALIDATION_{arguments.run_index:02d}"
    ).resolve()
    output = Path(arguments.output_dir).expanduser().resolve()
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
        raise PermissionError("MASTER_STATE does not authorize A1 validation")
    if state.get("phase") != "DYNAMIC_FIX_VALIDATION_RUNNING":
        raise PermissionError("A1 is not in fix-validation phase")
    if dynamic.get("current_task") != TASK_ID:
        raise PermissionError("dynamic task differs from A1")
    if dynamic.get("targeted_fix_count") != 1:
        raise PermissionError("the single targeted fix is not registered")
    if dynamic.get("validation_processes_started") != arguments.run_index:
        raise PermissionError("validation started counter does not match run index")
    if dynamic.get("validation_processes_completed") != arguments.run_index - 1:
        raise PermissionError("validation completion sequence is not contiguous")
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


def _run(arguments: argparse.Namespace, frozen: dict[str, Any], application: Any) -> dict[str, Any]:
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
        raise RuntimeError("failed to open frozen assembly-control model")
    for _ in range(3):
        application.update()
    p1 = frozen["acceptance"]["benches"]["P1"]
    profile = p1["inputs"]["component_driver_profile"]
    rate_hz = int(frozen["acceptance"]["shared_numeric_profile"]["physics_rate_hz"])
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / rate_hz,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    runtime_fix = configure_continuous_plug_guide_runtime_collision(
        stage, frozen["contract"]
    )
    timeline: dict[str, dict[str, Any]] = {}
    _set_initial_translation(stage, BODY_PATH, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, UsdGeom, Gf)
    timeline["T3"] = _stage_snapshot(
        "T3",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="post-fix immediately before World.reset",
    )
    for owner in OWNER_PATHS.values():
        prim = stage.GetPrimAtPath(owner)
        if not prim:
            raise RuntimeError(f"missing contact-report owner {owner}")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    world.get_physics_context().set_gravity(float(profile["gravity_magnitude_m_s2"]))
    world.reset()
    interface = get_physx_simulation_interface()
    timeline["T4"] = _stage_snapshot(
        "T4",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="post-fix immediately after World.reset",
    )
    contacts_t4 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T4")
    views = {
        label: RigidPrim(
            prim_paths_expr=path,
            name=f"validation_{arguments.run_index}_{label}",
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
        semantic="post-fix after RigidPrim view initialization",
    )
    world.step(render=False)
    timeline["T7"] = _view_snapshot(
        "T7",
        views,
        explicit_step_count=1,
        semantic="post-fix after first explicit physics step",
    )
    contacts_t7 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T7")

    errors = {
        stage_id: abs(float(timeline[stage_id]["datum_error_m"]))
        for stage_id in ("T4", "T5", "T7")
    }
    joint_limit = frozen["contract"]["coupling_nut_motion"]["transZ_backup_limits_m"]
    nut_relative_z = {
        stage_id: float(timeline[stage_id]["nut_body_relative_position_m"][2])
        for stage_id in ("T4", "T5", "T7")
    }
    datum_pass = all(value <= TOLERANCE_M for value in errors.values())
    nut_pass = all(
        float(joint_limit["low"]) <= value <= float(joint_limit["high"])
        for value in nut_relative_z.values()
    )
    fixed_t5 = timeline["T5"]["bodies"]["fixed_receptacle"]["position_m"]
    fixed_t7 = timeline["T7"]["bodies"]["fixed_receptacle"]["position_m"]
    fixed_drift = sum(
        (float(fixed_t7[index]) - float(fixed_t5[index])) ** 2 for index in range(3)
    ) ** 0.5
    contacts = _aggregate_contacts((contacts_t4, contacts_t7))
    passed = bool(datum_pass and nut_pass)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_index": arguments.run_index,
        "status": "PASS" if passed else "FAIL",
        "individual_validation_passed": passed,
        "timeline": timeline,
        "maximum_abs_datum_error_m": max(errors.values()),
        "datum_error_by_stage_m": errors,
        "datum_tolerance_m": TOLERANCE_M,
        "datum_pass": datum_pass,
        "nut_body_relative_z_by_stage_m": nut_relative_z,
        "nut_body_joint_limit_m": joint_limit,
        "nut_relative_position_pass": nut_pass,
        "fixed_receptacle_translation_drift_m": fixed_drift,
        "runtime_collision_fix": runtime_fix,
        "contact_statistics": contacts,
        "physics_rate_hz": rate_hz,
        "explicit_physics_step_count": 1,
        "object_pose_write_after_physics_start_count": 0,
        "solver_error_count": 0,
        "source_asset_written": False,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    world.stop()
    World.clear_instance()
    context.close_stage()
    application.update()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _authorize(arguments)
    output = frozen["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(
            tempfile.mkdtemp(
                prefix=f"kcg-dyn-a1-validation-{arguments.run_index:02d}-", dir="/tmp"
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
            "contract": _sha256(frozen["contract_path"]),
            "model": _sha256(frozen["model_path"]),
            "mapping": _sha256(frozen["mapping_path"]),
            "acceptance": _sha256(frozen["acceptance_path"]),
        }
        report["frozen_inputs_unchanged"] = report["post_run_sha256"] == EXPECTED_SHA256
        report["kit_portable_root"] = str(portable)
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen input SHA-256 changed during validation")
        exit_code = 0 if report["individual_validation_passed"] else 3
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_index": arguments.run_index,
            "status": "ERROR",
            "individual_validation_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "object_pose_write_after_physics_start_count": 0,
            "dynamic_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
            "source_asset_written": False,
        }
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
