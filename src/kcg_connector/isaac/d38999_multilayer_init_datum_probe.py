#!/usr/bin/env python3

"""A/B diagnose the multilayer model's post-reset datum offset.

This is diagnostic-only.  Both branches load the same frozen assembly-control
asset and use the same pre-reset pose.  The sole A/B change is whether all
connector colliders receive the already-authoritative fine-connector PhysX
contact/rest offsets on the in-memory stage before ``World.reset()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_init_datum_probe_v1"
TASK_ID = "EIGHT-HOUR-A1-INIT-DATUM"
START_SEPARATION_M = 0.00550
TOLERANCE_M = 0.00005
OLD_OBSERVED_ERROR_M = 0.00022602637112140687
CONTACT_OFFSET_M = 1.0e-5
REST_OFFSET_M = 0.0
EXPECTED_COLLIDER_COUNT = 19
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"
PLUG_GUIDE_PATH = BODY_PATH + "/MatingShell/ContinuousPlugGuide"
MODEL_RELATIVE = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
MASTER_RELATIVE = Path("src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
PHYSICAL_RELATIVE = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
OUTPUT_RELATIVE = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/DIAGNOSTIC_AB"
)
EXPECTED_SHA256 = {
    "model": "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec",
    "master_contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--physical-contract", required=True)
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
    if state.get("status") != "RUNNING" or state.get("task_id") != TASK_ID:
        raise PermissionError("MASTER_STATE does not authorize A1")
    if queue.get("status") != "RUNNING" or queue.get("current_task") != "A1":
        raise PermissionError("WORK_QUEUE does not authorize A1")
    master_path = _frozen_path(
        arguments.contract,
        MASTER_RELATIVE,
        EXPECTED_SHA256["master_contract"],
        "master contract",
    )
    physical_path = _frozen_path(
        arguments.physical_contract,
        PHYSICAL_RELATIVE,
        EXPECTED_SHA256["physical_contract"],
        "physical contract",
    )
    model_path = _frozen_path(
        arguments.model,
        MODEL_RELATIVE,
        EXPECTED_SHA256["model"],
        "assembly-control model",
    )
    master = yaml.safe_load(master_path.read_text(encoding="utf-8"))
    physical = yaml.safe_load(physical_path.read_text(encoding="utf-8"))
    inherited = master["authoritative_inputs"]["physical_model_contract"]
    if inherited["path"] != PHYSICAL_RELATIVE.as_posix():
        raise PermissionError("master contract physical-contract path changed")
    if inherited["sha256"] != EXPECTED_SHA256["physical_contract"]:
        raise PermissionError("master contract physical-contract SHA changed")
    runtime = physical["solver_profile"]
    if runtime["fine_connector_contact_offset_m"] != CONTACT_OFFSET_M:
        raise PermissionError("authoritative fine connector contact offset changed")
    if runtime["rest_offset_m"] != REST_OFFSET_M:
        raise PermissionError("authoritative rest offset changed")
    if (
        physical["a2_collision_authoring_blueprint"]["global"]
        ["automatic_contact_or_rest_offset_allowed"]
        is not False
    ):
        raise PermissionError("automatic contact/rest offsets became authorized")
    return {
        "output": output,
        "master_path": master_path,
        "physical_path": physical_path,
        "model_path": model_path,
        "input_sha256": dict(EXPECTED_SHA256),
    }


def _finite(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} is not a finite {size}-vector")
    return result


def _set_initial_translation(stage: Any, path: str, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing rigid body prim {path}")
    xformable = usd_geom.Xformable(prim)
    if xformable.GetOrderedXformOps():
        raise RuntimeError(f"unexpected authored transform stack at {path}")
    xformable.AddTranslateOp().Set(gf.Vec3d(0.0, 0.0, -START_SEPARATION_M))


def _stage_position(stage: Any, path: str, usd_geom: Any) -> list[float]:
    prim = stage.GetPrimAtPath(path)
    matrix = usd_geom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    return [float(translation[index]) for index in range(3)]


def _configure_dynamic_mesh(stage: Any) -> dict[str, Any]:
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
        "same_in_both_branches": True,
    }


def _collision_prims(stage: Any, usd_physics: Any) -> list[Any]:
    return [prim for prim in stage.Traverse() if prim.HasAPI(usd_physics.CollisionAPI)]


def _offset_inventory(prims: list[Any]) -> list[dict[str, Any]]:
    inventory = []
    for prim in prims:
        contact = prim.GetAttribute("physxCollision:contactOffset")
        rest = prim.GetAttribute("physxCollision:restOffset")
        inventory.append(
            {
                "path": str(prim.GetPath()),
                "contact_offset_m": None if not contact or not contact.HasAuthoredValueOpinion() else float(contact.Get()),
                "rest_offset_m": None if not rest or not rest.HasAuthoredValueOpinion() else float(rest.Get()),
            }
        )
    return inventory


def _apply_authoritative_offsets(prims: list[Any], physx_schema: Any) -> None:
    for prim in prims:
        api = physx_schema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
        api.CreateRestOffsetAttr().Set(REST_OFFSET_M)


def _contact_inventory(interface: Any, schema_tools: Any) -> list[dict[str, Any]]:
    headers, contacts, _friction = interface.get_full_contact_report()
    rows = []
    for header in headers:
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(contacts[index].separation) for index in range(start, stop)]
        rows.append(
            {
                "collider_paths": [
                    str(schema_tools.intToSdfPath(header.collider0)),
                    str(schema_tools.intToSdfPath(header.collider1)),
                ],
                "contact_record_count": int(header.num_contact_data),
                "minimum_separation_m": min(separations) if separations else None,
            }
        )
    return rows


def _run_case(
    *,
    case_id: str,
    explicit_offsets: bool,
    frozen: dict[str, Any],
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
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    cooking = _configure_dynamic_mesh(stage)
    collision_prims = _collision_prims(stage, UsdPhysics)
    if len(collision_prims) != EXPECTED_COLLIDER_COUNT:
        raise RuntimeError(
            f"collision count changed: {len(collision_prims)} != {EXPECTED_COLLIDER_COUNT}"
        )
    offsets_before = _offset_inventory(collision_prims)
    if explicit_offsets:
        _apply_authoritative_offsets(collision_prims, PhysxSchema)
    offsets_after = _offset_inventory(collision_prims)
    _set_initial_translation(stage, BODY_PATH, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, UsdGeom, Gf)
    for owner in (FIXED_PATH, BODY_PATH, NUT_PATH):
        prim = stage.GetPrimAtPath(owner)
        if not prim:
            raise RuntimeError(f"missing owner {owner}")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    pre_reset = {
        "fixed_receptacle_position_m": _stage_position(stage, FIXED_PATH, UsdGeom),
        "body_assembly_position_m": _stage_position(stage, BODY_PATH, UsdGeom),
        "coupling_nut_position_m": _stage_position(stage, NUT_PATH, UsdGeom),
    }
    pre_reset["datum_separation_m"] = (
        pre_reset["fixed_receptacle_position_m"][2]
        - pre_reset["body_assembly_position_m"][2]
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    views = {
        "fixed_receptacle": RigidPrim(
            prim_paths_expr=FIXED_PATH,
            name=f"{case_id}_fixed",
            reset_xform_properties=False,
        ),
        "body_assembly": RigidPrim(
            prim_paths_expr=BODY_PATH,
            name=f"{case_id}_body",
            reset_xform_properties=False,
        ),
        "coupling_nut": RigidPrim(
            prim_paths_expr=NUT_PATH,
            name=f"{case_id}_nut",
            reset_xform_properties=False,
        ),
    }
    for view in views.values():
        view.initialize()
    post_reset: dict[str, Any] = {}
    handles: dict[str, bool] = {}
    for label, view in views.items():
        positions, orientations = view.get_world_poses()
        position = _finite(positions[0], 3, label + " position")
        orientation = _finite(orientations[0], 4, label + " orientation")
        velocity = _finite(view.get_velocities()[0], 6, label + " velocity")
        post_reset[label] = {
            "position_m": position.tolist(),
            "orientation_wxyz": orientation.tolist(),
            "velocity": velocity.tolist(),
        }
        handles[label] = bool(view.is_physics_handle_valid())
    datum_separation = float(
        post_reset["fixed_receptacle"]["position_m"][2]
        - post_reset["body_assembly"]["position_m"][2]
    )
    nut_body_relative_z = float(
        post_reset["coupling_nut"]["position_m"][2]
        - post_reset["body_assembly"]["position_m"][2]
    )
    interface = get_physx_simulation_interface()
    contacts = _contact_inventory(interface, PhysicsSchemaTools)
    result = {
        "case_id": case_id,
        "explicit_authoritative_offsets": explicit_offsets,
        "runtime_collision_cooking": cooking,
        "collision_prim_count": len(collision_prims),
        "offsets_before": offsets_before,
        "offsets_after": offsets_after,
        "pre_reset": pre_reset,
        "post_reset": post_reset,
        "physics_handles": handles,
        "post_reset_datum_separation_m": datum_separation,
        "post_reset_datum_error_m": datum_separation - START_SEPARATION_M,
        "post_reset_nut_body_relative_z_m": nut_body_relative_z,
        "contact_pair_count_after_reset": len(contacts),
        "contacts_after_reset": contacts,
        "explicit_world_step_count": 0,
        "object_pose_write_after_physics_start_count": 0,
        "source_asset_written": False,
    }
    world.stop()
    World.clear_instance()
    return result


def _diagnose(frozen: dict[str, Any], application: Any) -> dict[str, Any]:
    baseline = _run_case(
        case_id="A_IMPLICIT_AUTOMATIC_OFFSETS",
        explicit_offsets=False,
        frozen=frozen,
        application=application,
    )
    explicit = _run_case(
        case_id="B_EXPLICIT_FINE_CONNECTOR_OFFSETS",
        explicit_offsets=True,
        frozen=frozen,
        application=application,
    )
    baseline_error = abs(float(baseline["post_reset_datum_error_m"]))
    explicit_error = abs(float(explicit["post_reset_datum_error_m"]))
    baseline_reproduced = baseline_error > TOLERANCE_M
    explicit_pass = explicit_error <= TOLERANCE_M
    if baseline_reproduced and explicit_pass:
        classification = "EXPLICIT_FROZEN_CONTACT_OFFSETS_CONFIRMED_ROOT_CAUSE"
    elif baseline_reproduced and not explicit_pass:
        classification = "CONTACT_OFFSET_HYPOTHESIS_REJECTED"
    else:
        classification = "DIAGNOSTIC_REPLAY_INCONSISTENT"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "COMPLETED",
        "classification": classification,
        "hypothesis_id": "A1-H1-EXPLICIT-CONTACT-OFFSETS",
        "baseline": baseline,
        "explicit_offset_case": explicit,
        "comparison": {
            "baseline_reproduced": baseline_reproduced,
            "baseline_error_m": baseline_error,
            "old_observed_error_m": OLD_OBSERVED_ERROR_M,
            "explicit_error_m": explicit_error,
            "improvement_m": baseline_error - explicit_error,
            "explicit_error_within_50_um": explicit_pass,
            "tolerance_m": TOLERANCE_M,
        },
        "diagnostic_only": True,
        "formal_nominal_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
        "source_asset_written": False,
        "collision_disabled": False,
        "explicit_physics_step_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _authorize(arguments)
    output = frozen["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-a1-init-datum-", dir="/tmp"))
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
            "model": _sha256(frozen["model_path"]),
            "master_contract": _sha256(frozen["master_path"]),
            "physical_contract": _sha256(frozen["physical_path"]),
        }
        report["frozen_inputs_unchanged"] = report["post_run_sha256"] == EXPECTED_SHA256
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": "ERROR",
            "classification": "DIAGNOSTIC_PROGRAM_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "diagnostic_only": True,
            "formal_nominal_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
            "source_asset_written": False,
        }
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
