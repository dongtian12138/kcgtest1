#!/usr/bin/env python3

"""Read the resolved B-scene solver contract without running a grasp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any


TASK_ID = "DYN-B-GRASP-LIFT-RECOVERY-V2"
HYPOTHESIS_ID = "B-V2-H9-ARM-DRIVE-NYQUIST-OSCILLATION"
RUN_ID = "B-V2-H9-SOLVER-CONTRACT-DIAGNOSTIC-01-IFIX01"

EXPECTED_SHA256 = {
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
    "assembly_control": "a3e43d53150dc94f1c703e41bcc6facd7df0f55ea7e083f8debf600349e8cc3d",
    "visual_complete": "69fe6dc3ca9caace8bb26cd0cfad68c0eb84111f09697da6068cd91802d65c0a",
    "local_contact_reference": "94b9cea0a7bb1e4d4a7c6583819abe1c722e252ae45012ba78f1a396b0a5ab85",
    "model_mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
}

RELATIVE_PATHS = {
    "physical_contract": Path(
        "src/kcg_connector/config/"
        "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
    ),
    "assembly_control": Path(
        "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "D38999_ASSEMBLY_CONTROL_V1.usda"
    ),
    "visual_complete": Path(
        "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "D38999_VISUAL_COMPLETE_V1.usda"
    ),
    "local_contact_reference": Path(
        "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "D38999_LOCAL_CONTACT_REFERENCE_V1.usda"
    ),
    "model_mapping": Path(
        "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json"
    ),
    "robot": Path(
        "artifacts/kcg_connector/isaac/robot/"
        "handarm_keyed_v3_physical_r7/handarm.usda"
    ),
}

ROBOT_ROOT = "/World/HandArm"
ROBOT_ARTICULATION = "/World/HandArm/Geometry/world"
CONNECTOR_REFERENCE_ROOT = "/World/D38999TabletopV1/D38999Pair"
CONNECTOR_MODEL_ROOT = (
    CONNECTOR_REFERENCE_ROOT
    + "/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
)
BODY_PATH = CONNECTOR_MODEL_ROOT + "/D38999Pair/LoosePlug/BodyAssembly"
NUT_PATH = CONNECTOR_MODEL_ROOT + "/D38999Pair/LoosePlug/CouplingNut"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _attribute(prim, name: str) -> dict[str, Any]:
    attribute = prim.GetAttribute(name)
    if not attribute.IsValid():
        return {
            "valid": False,
            "authored": False,
            "value": None,
        }
    return {
        "valid": True,
        "authored": bool(attribute.HasAuthoredValueOpinion()),
        "value": _json_value(attribute.Get()),
        "type_name": str(attribute.GetTypeName()),
    }


def _matching_attributes(prim) -> dict[str, Any]:
    tokens = (
        "solver",
        "iteration",
        "ccd",
        "stabil",
        "determin",
        "externalforces",
        "timestep",
        "depenetration",
    )
    result = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if any(token in name.lower() for token in tokens):
            result[name] = {
                "authored": bool(attribute.HasAuthoredValueOpinion()),
                "value": _json_value(attribute.Get()),
                "type_name": str(attribute.GetTypeName()),
            }
    return result


def _scalar(value: Any) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return int(value)


def _run(repository: Path) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    World.clear_instance()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    add_reference_to_stage(str(repository / RELATIVE_PATHS["robot"]), ROBOT_ROOT)
    add_reference_to_stage(
        str(repository / RELATIVE_PATHS["assembly_control"]),
        CONNECTOR_REFERENCE_ROOT,
    )
    robot = world.scene.add(
        SingleArticulation(
            prim_path=ROBOT_ARTICULATION,
            name="solver_contract_robot",
        )
    )
    body = world.scene.add(
        SingleRigidPrim(prim_path=BODY_PATH, name="solver_contract_body")
    )
    nut = world.scene.add(
        SingleRigidPrim(prim_path=NUT_PATH, name="solver_contract_nut")
    )
    scene_path = str(world.get_physics_context().prim_path)
    scene_prim = stage.GetPrimAtPath(scene_path)
    before_reset = {
        "scene": _matching_attributes(scene_prim),
        "robot": _matching_attributes(stage.GetPrimAtPath(ROBOT_ARTICULATION)),
        "body": _matching_attributes(stage.GetPrimAtPath(BODY_PATH)),
        "nut": _matching_attributes(stage.GetPrimAtPath(NUT_PATH)),
    }
    world.reset()
    if not robot.handles_initialized:
        raise RuntimeError("robot articulation handles were not initialized")
    robot_position_iterations = int(robot.get_solver_position_iteration_count())
    robot_velocity_iterations = int(robot.get_solver_velocity_iteration_count())
    body_position_iterations = _scalar(
        body._rigid_prim_view.get_solver_position_iteration_counts()
    )
    body_velocity_iterations = _scalar(
        body._rigid_prim_view.get_solver_velocity_iteration_counts()
    )
    nut_position_iterations = _scalar(
        nut._rigid_prim_view.get_solver_position_iteration_counts()
    )
    nut_velocity_iterations = _scalar(
        nut._rigid_prim_view.get_solver_velocity_iteration_counts()
    )
    after_reset = {
        "scene": _matching_attributes(scene_prim),
        "robot": _matching_attributes(stage.GetPrimAtPath(ROBOT_ARTICULATION)),
        "body": _matching_attributes(stage.GetPrimAtPath(BODY_PATH)),
        "nut": _matching_attributes(stage.GetPrimAtPath(NUT_PATH)),
    }
    selected_scene_attributes = {
        name: _attribute(scene_prim, name)
        for name in (
            "physxScene:solverType",
            "physxScene:enableCCD",
            "physxScene:enableStabilization",
            "physxScene:enableEnhancedDeterminism",
            "physxScene:enableExternalForcesEveryIteration",
            "physxScene:minPositionIterationCount",
            "physxScene:minVelocityIterationCount",
            "physxScene:timeStepsPerSecond",
        )
    }
    world.stop()
    resolved = {
        "physics_scene_prim_path": scene_path,
        "world_physics_dt_s": float(world.get_physics_dt()),
        "robot_articulation": {
            "solver_position_iteration_count": robot_position_iterations,
            "solver_velocity_iteration_count": robot_velocity_iterations,
        },
        "plug_body": {
            "solver_position_iteration_count": body_position_iterations,
            "solver_velocity_iteration_count": body_velocity_iterations,
        },
        "coupling_nut": {
            "solver_position_iteration_count": nut_position_iterations,
            "solver_velocity_iteration_count": nut_velocity_iterations,
        },
    }
    matches = {
        "physics_dt": abs(resolved["world_physics_dt_s"] - 1.0 / 240.0)
        <= 1.0e-15,
        "robot_iterations": (
            robot_position_iterations == 32 and robot_velocity_iterations == 8
        ),
        "body_iterations": (
            body_position_iterations == 32 and body_velocity_iterations == 8
        ),
        "nut_iterations": (
            nut_position_iterations == 32 and nut_velocity_iterations == 8
        ),
    }
    return {
        "schema_version": "d38999_b_solver_contract_probe_v1",
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "run_id": RUN_ID,
        "status": "DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "formal_dynamic_pass_claimed": False,
        "executed_explicit_physics_step_count": 0,
        "object_pose_write_after_physics_start_count": 0,
        "control_action_count": 0,
        "expected": {
            "physics_rate_hz": 240,
            "physics_dt_s": 1.0 / 240.0,
            "solver_type": "TGS",
            "position_iterations": 32,
            "velocity_iterations": 8,
            "substeps": 1,
            "stabilization_enabled": True,
        },
        "resolved": resolved,
        "selected_scene_attributes": selected_scene_attributes,
        "all_matching_attributes_before_reset": before_reset,
        "all_matching_attributes_after_reset": after_reset,
        "contract_match": matches,
        "all_numeric_contract_values_match": all(matches.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root")
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {name: repository / relative for name, relative in RELATIVE_PATHS.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    pre_sha256 = {name: _sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected, "actual": pre_sha256[name]}
        for name, expected in EXPECTED_SHA256.items()
        if pre_sha256[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen input hash mismatch: {mismatches}")
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-b-h9-solver-", dir="/tmp"))
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
        report = _run(repository)
        report["pre_run_sha256"] = pre_sha256
        report["post_run_sha256"] = {
            name: _sha256(path) for name, path in paths.items()
        }
        report["frozen_inputs_unchanged"] = (
            report["pre_run_sha256"] == report["post_run_sha256"]
        )
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen inputs changed during diagnostic")
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": "d38999_b_solver_contract_probe_v1",
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "run_id": RUN_ID,
            "status": "DIAGNOSTIC_PROGRAM_ERROR",
            "diagnostic_only": True,
            "formal_dynamic_pass_claimed": False,
            "executed_explicit_physics_step_count": 0,
            "object_pose_write_after_physics_start_count": 0,
            "control_action_count": 0,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
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
        application.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
