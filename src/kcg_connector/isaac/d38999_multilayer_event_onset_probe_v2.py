#!/usr/bin/env python3

"""Run exactly one bounded, local D38999 V2 event-onset probe in Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


TASK_ID = "DYN-A1-EVENT-ONSET-CALIBRATION-V2"
HYPOTHESIS_ID = "A1-V2-H2-LOCAL-EVENT-ONSET-DECOMPOSITION"
EVENT04_FIX_HYPOTHESIS_ID = "A1-V2-H3-EVENT04-LATERAL-DISCRETE-STABILITY"
EVENT04_PASSIVITY_HYPOTHESIS_ID = "A1-V2-H4-EVENT04-PASSIVITY-EVALUATOR"
SCHEMA_VERSION = "kcg_d38999_multilayer_event_onset_probe_v2"
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"

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
INITIALIZATION_AGGREGATE_RELATIVE = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/INIT_GATE_AGGREGATE_RESULT.json"
)
OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/EVENT_PROBES"
)

EXPECTED_SHA256 = {
    "master_contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
    "acceptance_contract": "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
    "authorized_overrides": "392766e8eceb85a3c910b118c2ad998aef891a74e58c31cd94e383c9908535ce",
    "model": "d5bcc5e8b28e31912f65cd87a0bbe5d7a035744f7f7d8c7b785e17cdad382a6e",
    "mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
    "fine_offset_result": "baee361e50a3a8db8109b0a809ae88a32de4907f35e9e96652c2e00d9b186dd6",
    "initialization_aggregate": "a5a586ca08e5a203b1f6b8770500128a7e45a9f97952c8b74dd46f125dc93545",
}

EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)

PROBE_HALF_WINDOW_M = 0.00025
PROFILE_DURATION_S = 2.0
HOLD_DURATION_S = 1.0
NUMERICAL_FORCE_FLOOR_N = 1.0e-9

EVENT_SPECS: dict[str, dict[str, Any]] = {
    "five_key_polarization": {
        "ordinal": 1,
        "mode": "physical_contact",
        "initial_x_m": 0.000130,
        "initial_yaw_deg": 0.30,
        "official_contact_family": (
            "continuous_keyway_wall <-> continuous_polarizing_key"
        ),
        "secondary_contact_family": (
            "continuous_shell_and_guidance <-> continuous_shell_and_guidance"
        ),
        "fixed_target_roles": {
            "continuous_keyway_wall",
            "continuous_shell_and_guidance",
        },
        "body_target_roles": {
            "continuous_polarizing_key",
            "continuous_shell_and_guidance",
        },
        "selected_criterion": "physical_force_onset",
    },
    "three_start_thread_entry": {
        "ordinal": 2,
        "mode": "equivalent_internal_action",
        "initial_x_m": 0.0,
        "initial_yaw_deg": 0.0,
        "fixed_target_roles": set(),
        "body_target_roles": set(),
        "selected_criterion": "constraint_force_onset",
    },
    "spring_finger_engagement": {
        "ordinal": 3,
        "mode": "equivalent_internal_action",
        "initial_x_m": 0.000020,
        "initial_yaw_deg": 0.0,
        "fixed_target_roles": set(),
        "body_target_roles": set(),
        "selected_criterion": "restoring_force_onset",
    },
    "first_pin_socket_spring_touch": {
        "ordinal": 4,
        "mode": "equivalent_internal_action",
        "initial_x_m": 0.000002,
        "initial_yaw_deg": 0.0,
        "fixed_target_roles": set(),
        "body_target_roles": set(),
        "selected_criterion": "same_label_restoring_force_onset",
    },
    "pin_barrier_seal_contact": {
        "ordinal": 5,
        "mode": "equivalent_internal_action",
        "initial_x_m": 0.0,
        "initial_yaw_deg": 0.0,
        "fixed_target_roles": set(),
        "body_target_roles": set(),
        "selected_criterion": "axial_resistance_force_onset",
    },
    "seal_compression": {
        "ordinal": 6,
        "mode": "equivalent_internal_action",
        "initial_x_m": 0.0,
        "initial_yaw_deg": 0.0,
        "fixed_target_roles": set(),
        "body_target_roles": set(),
        "selected_criterion": "annular_axial_force_onset",
    },
    "shell_to_shell_metal_bottoming": {
        "ordinal": 7,
        "mode": "physical_contact",
        "initial_x_m": 0.0,
        "initial_yaw_deg": 0.0,
        "official_contact_family": (
            "continuous_real_metal_stop_fixed <-> "
            "continuous_real_metal_stop_plug"
        ),
        "fixed_target_roles": {"continuous_real_metal_stop_fixed"},
        "body_target_roles": {"continuous_real_metal_stop_plug"},
        "selected_criterion": "physical_force_onset",
    },
}


def _run_hypothesis_id(event: str, validation_attempt: int) -> str:
    if event == "first_pin_socket_spring_touch" and validation_attempt == 2:
        return EVENT04_PASSIVITY_HYPOTHESIS_ID
    if event == "first_pin_socket_spring_touch" and validation_attempt == 1:
        return EVENT04_FIX_HYPOTHESIS_ID
    return HYPOTHESIS_ID


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=EVENT_ORDER)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--physical-contract", required=True)
    parser.add_argument("--acceptance-contract", required=True)
    parser.add_argument("--authorized-overrides", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--fine-offset-result", required=True)
    parser.add_argument("--initialization-aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    parser.add_argument("--validation-attempt", type=int, choices=(0, 1, 2), default=0)
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
    spec = EVENT_SPECS[arguments.event]
    ordinal = int(spec["ordinal"])
    output = Path(arguments.output_dir).expanduser().resolve()
    suffix = (
        ""
        if arguments.validation_attempt == 0
        else f"_VALIDATION_{arguments.validation_attempt:02d}"
    )
    expected_output = (
        repository
        / OUTPUT_ROOT_RELATIVE
        / f"EVENT_{ordinal:02d}_{arguments.event}{suffix}"
    ).resolve()
    if output != expected_output:
        raise PermissionError(f"output path is frozen: {output} != {expected_output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite event evidence: {output}")

    path_inputs = (
        ("master_contract", arguments.contract, MASTER_RELATIVE),
        ("physical_contract", arguments.physical_contract, PHYSICAL_RELATIVE),
        ("acceptance_contract", arguments.acceptance_contract, ACCEPTANCE_RELATIVE),
        ("authorized_overrides", arguments.authorized_overrides, OVERRIDES_RELATIVE),
        ("model", arguments.model, MODEL_RELATIVE),
        ("mapping", arguments.mapping, MAPPING_RELATIVE),
        ("fine_offset_result", arguments.fine_offset_result, FINE_OFFSET_RESULT_RELATIVE),
        (
            "initialization_aggregate",
            arguments.initialization_aggregate,
            INITIALIZATION_AGGREGATE_RELATIVE,
        ),
    )
    paths = {
        label: _frozen_path(raw, relative, EXPECTED_SHA256[label], label)
        for label, raw, relative in path_inputs
    }
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
    initialization = v2.get("initialization_gate", {})
    probes = v2.get("event_onset_probes", {})
    required = {
        "root_task": (state.get("task_id"), "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2"),
        "root_status": (state.get("status"), "VALIDATING"),
        "phase": (state.get("phase"), "DYN_A1_EVENT_ONSET_CALIBRATION_V2"),
        "current_node": (v2.get("current_node"), TASK_ID),
        "initialization_status": (initialization.get("status"), "DYNAMIC_PASS"),
        "initialization_pass_count": (initialization.get("processes_passed"), 3),
        "probe_status": (probes.get("status"), "VALIDATING"),
        "probe_current_event": (probes.get("current_event"), arguments.event),
        "queue_status": (queue.get("status"), "VALIDATING"),
        "queue_task": (queue.get("current_task"), TASK_ID),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    started = list(probes.get("events_started", []))
    completed = list(probes.get("events_completed", []))
    if started != list(EVENT_ORDER[:ordinal]):
        mismatches["events_started"] = {
            "actual": started,
            "expected": list(EVENT_ORDER[:ordinal]),
        }
    if completed != list(EVENT_ORDER[: ordinal - 1]):
        mismatches["events_completed"] = {
            "actual": completed,
            "expected": list(EVENT_ORDER[: ordinal - 1]),
        }
    if arguments.validation_attempt:
        authorization = probes.get("targeted_fix_validation", {})
        expected_authorization = {
            "event": arguments.event,
            "attempt": arguments.validation_attempt,
            "status": "REGISTERED",
        }
        actual_authorization = {
            key: authorization.get(key) for key in expected_authorization
        }
        if actual_authorization != expected_authorization:
            mismatches["targeted_fix_validation"] = {
                "actual": actual_authorization,
                "expected": expected_authorization,
            }
    elif arguments.event in list(probes.get("diagnostic_runs_completed", [])):
        mismatches["validation_attempt"] = {
            "actual": 0,
            "expected": "nonzero because the immutable diagnostic run already completed",
        }
    if mismatches:
        raise PermissionError(f"V2 event probe state guard failed: {mismatches}")

    master = yaml.safe_load(paths["master_contract"].read_text(encoding="utf-8"))
    physical = yaml.safe_load(paths["physical_contract"].read_text(encoding="utf-8"))
    acceptance = yaml.safe_load(paths["acceptance_contract"].read_text(encoding="utf-8"))
    aggregate = json.loads(paths["initialization_aggregate"].read_text(encoding="utf-8"))
    if aggregate.get("classification") != "V2_INITIALIZATION_GATE_3_OF_3_PASS":
        raise PermissionError("initialization aggregate is not the required 3/3 pass")
    event_rows = master["assembly_events"]["ordered"]
    if [row["name"] for row in event_rows] != list(EVENT_ORDER):
        raise PermissionError("master event order changed")
    nominal_m = float(event_rows[ordinal - 1]["nominal_separation_m"])
    p1 = acceptance["benches"]["P1"]
    shared = acceptance["shared_numeric_profile"]
    profile = p1["inputs"]["component_driver_profile"]
    thresholds = {
        "event_position_tolerance_m": float(
            p1["pass"]["nominal_event_position_tolerance_m"]
        ),
        "force_component_limit_n": float(profile["translation_force_component_limit_n"]),
        "torque_component_limit_nm": float(profile["torque_component_limit_nm"]),
        "fixed_drift_limit_m": float(
            master["acceptance_limits"]["maximum_fixed_receptacle_translation_drift_m"]
        ),
        "hard_penetration_limit_m": float(
            master["acceptance_limits"]["maximum_noncompliant_hard_penetration_m"]
        ),
        "contact_offset_m": float(shared["fine_contact_offset_m"]),
        "rest_offset_m": float(shared["rest_offset_m"]),
        "physics_rate_hz": int(shared["physics_rate_hz"]),
        "axial_speed_limit_m_s": float(p1["inputs"]["axial_speed_m_s"]),
        "gravity_m_s2": float(profile["gravity_magnitude_m_s2"]),
    }
    expected_thresholds = {
        "event_position_tolerance_m": 5.0e-05,
        "force_component_limit_n": 8.0,
        "torque_component_limit_nm": 0.30,
        "fixed_drift_limit_m": 5.0e-06,
        "hard_penetration_limit_m": 5.0e-05,
        "contact_offset_m": 1.0e-05,
        "rest_offset_m": 0.0,
        "physics_rate_hz": 240,
        "axial_speed_limit_m_s": 0.0005,
        "gravity_m_s2": 0.0,
    }
    if thresholds != expected_thresholds:
        raise PermissionError(f"frozen event thresholds changed: {thresholds}")
    if physical["solver_profile"]["fine_connector_contact_offset_m"] != 1.0e-05:
        raise PermissionError("physical contact offset changed")
    scale = float(master["contact_layout"]["coordinate_scale_m_per_in"])
    pairs = [
        {
            "label": str(row["label"]),
            "center_m": np.asarray(row["center_in"], dtype=np.float64) * scale,
            "same_label_only": True,
        }
        for row in master["contact_layout"]["pairs"]
    ]
    if len(pairs) != 61 or len({row["label"] for row in pairs}) != 61:
        raise PermissionError("61 unique same-label pairs are not present")
    return {
        "output": output,
        "paths": paths,
        "master": master,
        "acceptance": acceptance,
        "thresholds": thresholds,
        "nominal_m": nominal_m,
        "spec": spec,
        "pairs": pairs,
        "validation_attempt": int(arguments.validation_attempt),
    }


def _finite(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} is not a finite {size}-vector")
    return result


def _yaw_wxyz(quaternion: Any) -> float:
    w, x, y, z = _finite(quaternion, 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise RuntimeError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _minimum_jerk(progress: float) -> tuple[float, float]:
    u = max(0.0, min(1.0, float(progress)))
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    derivative = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    return position, derivative


def _set_initial_pose(
    stage: Any,
    path: str,
    *,
    x_m: float,
    z_m: float,
    yaw_deg: float,
    usd_geom: Any,
    gf: Any,
) -> None:
    prim = stage.GetPrimAtPath(path)
    xformable = usd_geom.Xformable(prim)
    if not prim or xformable.GetOrderedXformOps():
        raise RuntimeError(f"unexpected initial transform stack: {path}")
    xformable.AddTranslateOp().Set(gf.Vec3d(x_m, 0.0, z_m))
    if yaw_deg:
        xformable.AddRotateZOp().Set(float(yaw_deg))


def _collision_inventory(stage: Any, usd_physics: Any, physx_schema: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        role_attr = prim.GetAttribute("kcg:collisionRole")
        trace_attr = prim.GetAttribute("kcg:traceLabel")
        contact = prim.GetAttribute("physxCollision:contactOffset")
        rest = prim.GetAttribute("physxCollision:restOffset")
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
                "role": str(role_attr.Get()) if role_attr and role_attr.Get() else "UNLABELED",
                "trace_label": (
                    str(trace_attr.Get())
                    if trace_attr and trace_attr.Get() is not None
                    else None
                ),
                "physx_api": bool(prim.HasAPI(physx_schema.PhysxCollisionAPI)),
                "contact_offset_m": float(contact.Get()) if contact and contact.Get() is not None else None,
                "rest_offset_m": float(rest.Get()) if rest and rest.Get() is not None else None,
            }
        )
    rows.sort(key=lambda row: row["path"])
    return rows


def _trace_index(trace_label: Any, prefix: str) -> int | None:
    if not isinstance(trace_label, str):
        return None
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", trace_label)
    return None if match is None else int(match.group(1))


def _key_labels_correspond(roles: Sequence[str], labels: Sequence[Any]) -> bool | None:
    if set(roles) != {"continuous_keyway_wall", "continuous_polarizing_key"}:
        return None
    keyway_indices = [
        _trace_index(label, "keyway")
        for role, label in zip(roles, labels)
        if role == "continuous_keyway_wall"
    ]
    key_indices = [
        _trace_index(label, "key")
        for role, label in zip(roles, labels)
        if role == "continuous_polarizing_key"
    ]
    if len(keyway_indices) != 1 or len(key_indices) != 1:
        return False
    return keyway_indices[0] is not None and keyway_indices[0] == key_indices[0]


def _probe_collision_partition(
    rows: Sequence[Mapping[str, Any]], event: str
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    spec = EVENT_SPECS[event]
    members: dict[str, list[str]] = {}
    designated: set[str] = set()
    allowed: list[tuple[str, str]] = []

    if event == "five_key_polarization":
        members["FixedGuide"] = [
            str(row["path"])
            for row in rows
            if row["owner"] == "fixed_receptacle"
            and row["role"] == "continuous_shell_and_guidance"
        ]
        members["BodyGuide"] = [
            str(row["path"])
            for row in rows
            if row["owner"] == "body_assembly"
            and row["role"] == "continuous_shell_and_guidance"
        ]
        allowed.append(("FixedGuide", "BodyGuide"))
        for index in range(5):
            fixed_name = f"FixedKey{index}"
            body_name = f"BodyKey{index}"
            members[fixed_name] = [
                str(row["path"])
                for row in rows
                if row["owner"] == "fixed_receptacle"
                and row["role"] == "continuous_keyway_wall"
                and _trace_index(row.get("trace_label"), "keyway") == index
            ]
            members[body_name] = [
                str(row["path"])
                for row in rows
                if row["owner"] == "body_assembly"
                and row["role"] == "continuous_polarizing_key"
                and _trace_index(row.get("trace_label"), "key") == index
            ]
            if len(members[fixed_name]) != 2 or len(members[body_name]) != 1:
                raise RuntimeError(
                    f"event1 key label partition changed at {index}: "
                    f"fixed={len(members[fixed_name])} body={len(members[body_name])}"
                )
            allowed.append((fixed_name, body_name))
        if not members["FixedGuide"] or not members["BodyGuide"]:
            raise RuntimeError("event1 continuous guide groups are empty")
        designated.update(path for values in members.values() for path in values)
    else:
        fixed_roles = set(spec["fixed_target_roles"])
        body_roles = set(spec["body_target_roles"])
        fixed_target = [
            str(row["path"])
            for row in rows
            if row["owner"] == "fixed_receptacle" and row["role"] in fixed_roles
        ]
        body_target = [
            str(row["path"])
            for row in rows
            if row["owner"] == "body_assembly" and row["role"] in body_roles
        ]
        if fixed_target:
            members["FixedTarget"] = fixed_target
        if body_target:
            members["BodyTarget"] = body_target
        designated.update(fixed_target)
        designated.update(body_target)
        if spec["mode"] == "physical_contact":
            if not fixed_target or not body_target:
                raise RuntimeError("physical event target collider groups are empty")
            allowed.append(("FixedTarget", "BodyTarget"))
        elif fixed_target or body_target:
            raise RuntimeError("internal event unexpectedly has physical target colliders")

    members["FixedOther"] = [
        str(row["path"])
        for row in rows
        if row["owner"] == "fixed_receptacle" and str(row["path"]) not in designated
    ]
    members["BodyOther"] = [
        str(row["path"])
        for row in rows
        if row["owner"] == "body_assembly" and str(row["path"]) not in designated
    ]
    members["Nut"] = [
        str(row["path"]) for row in rows if row["owner"] == "coupling_nut"
    ]
    members = {name: values for name, values in members.items() if values}
    covered = sorted(path for values in members.values() for path in values)
    expected = sorted(str(row["path"]) for row in rows)
    if covered != expected or len(covered) != len(set(covered)):
        raise RuntimeError("event collision groups do not partition active colliders")
    return members, allowed


def _author_probe_collision_filter(
    stage: Any,
    rows: Sequence[Mapping[str, Any]],
    event: str,
    usd_geom: Any,
    usd_physics: Any,
    sdf: Any,
) -> dict[str, Any]:
    spec = EVENT_SPECS[event]
    fixed_target_roles = set(spec["fixed_target_roles"])
    body_target_roles = set(spec["body_target_roles"])
    members, allowed_group_pairs = _probe_collision_partition(rows, event)
    root = "/World/D38999V2EventProbeCollisionGroups"
    usd_geom.Scope.Define(stage, root)
    groups: dict[str, Any] = {}
    paths: dict[str, str] = {}
    for name, values in members.items():
        if not values:
            continue
        path = root + "/" + name
        group = usd_physics.CollisionGroup.Define(stage, path)
        group.CreateInvertFilteredGroupsAttr(False)
        collection = group.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr("explicitOnly")
        collection.CreateIncludeRootAttr(False)
        collection.CreateIncludesRel().SetTargets([sdf.Path(value) for value in values])
        authored = sorted(str(value) for value in collection.GetIncludesRel().GetTargets())
        if authored != sorted(values):
            raise RuntimeError(f"collision group readback failed: {name}")
        groups[name] = group
        paths[name] = path

    fixed_groups = sorted(name for name in groups if name.startswith("Fixed"))
    body_groups = sorted(name for name in groups if name.startswith("Body"))
    allowed_set = set(allowed_group_pairs)
    requested_filters = [
        (fixed_name, body_name)
        for fixed_name in fixed_groups
        for body_name in body_groups
        if (fixed_name, body_name) not in allowed_set
    ]
    requested_filters.extend(("Nut", name) for name in fixed_groups + body_groups)
    authored_filters: list[list[str]] = []
    for left, right in requested_filters:
        if left not in groups or right not in groups:
            continue
        groups[left].CreateFilteredGroupsRel().AddTarget(sdf.Path(paths[right]))
        authored_filters.append([left, right])
    target_pair_enabled = bool(allowed_group_pairs) and all(
        left in groups and right in groups for left, right in allowed_group_pairs
    )
    if spec["mode"] == "physical_contact" and not target_pair_enabled:
        raise RuntimeError("physical event target collider groups are empty")
    if spec["mode"] != "physical_contact" and target_pair_enabled:
        raise RuntimeError("internal event unexpectedly enabled a physical target pair")
    covered = sorted(path for values in members.values() for path in values)
    expected = sorted(str(row["path"]) for row in rows)
    if covered != expected or len(covered) != len(set(covered)):
        raise RuntimeError("event collision groups do not partition active colliders")
    return {
        "authored_before_physics": True,
        "controller_input": False,
        "mode": "predetermined_role_pair_isolation_without_runtime_contact_truth",
        "member_counts": {name: len(values) for name, values in members.items()},
        "filtered_group_pairs": authored_filters,
        "allowed_group_pairs": [list(pair) for pair in allowed_group_pairs],
        "same_index_key_pairing_enforced": event == "five_key_polarization",
        "target_pair_enabled": target_pair_enabled,
        "allowed_target_roles": {
            "fixed": sorted(fixed_target_roles),
            "body": sorted(body_target_roles),
        },
    }


def _contact_metrics(stage: Any, interface: Any, schema_tools: Any, dt: float) -> dict[str, Any]:
    headers, contacts, _friction = interface.get_full_contact_report()
    families: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for header in headers:
        actors = [
            str(schema_tools.intToSdfPath(header.actor0)),
            str(schema_tools.intToSdfPath(header.actor1)),
        ]
        fixed = any(path.startswith(FIXED_PATH) for path in actors)
        loose = any(path.startswith(BODY_PATH) or path.startswith(NUT_PATH) for path in actors)
        if not (fixed and loose):
            continue
        colliders = [
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        ]
        roles = []
        trace_labels = []
        for path in colliders:
            prim = stage.GetPrimAtPath(path)
            attr = prim.GetAttribute("kcg:collisionRole") if prim else None
            roles.append(str(attr.Get()) if attr and attr.Get() is not None else "UNLABELED")
            trace_attr = prim.GetAttribute("kcg:traceLabel") if prim else None
            trace_labels.append(
                str(trace_attr.Get())
                if trace_attr and trace_attr.Get() is not None
                else None
            )
        family = " <-> ".join(sorted(roles))
        corresponding_key_label = _key_labels_correspond(roles, trace_labels)
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
        row = {
            "family": family,
            "collider_paths": colliders,
            "trace_labels": trace_labels,
            "corresponding_key_label": corresponding_key_label,
            "record_count": len(records),
            "minimum_separation_m": min(
                (record["separation_m"] for record in records), default=None
            ),
            "maximum_impulse_norm_n_s": max(
                (record["impulse_norm_n_s"] for record in records), default=0.0
            ),
            "maximum_equivalent_force_n": max(
                (record["impulse_norm_n_s"] / dt for record in records), default=0.0
            ),
            "strongest_record": max(
                records, key=lambda record: record["impulse_norm_n_s"], default=None
            ),
        }
        all_rows.append(row)
        aggregate = families.setdefault(
            family,
            {
                "active_pair_count": 0,
                "record_count": 0,
                "minimum_separation_m": None,
                "maximum_impulse_norm_n_s": 0.0,
                "maximum_equivalent_force_n": 0.0,
                "representative_collider_paths": colliders,
                "strongest_record": None,
            },
        )
        aggregate["active_pair_count"] += 1
        aggregate["record_count"] += len(records)
        separation = row["minimum_separation_m"]
        if separation is not None and (
            aggregate["minimum_separation_m"] is None
            or separation < aggregate["minimum_separation_m"]
        ):
            aggregate["minimum_separation_m"] = separation
        if row["maximum_impulse_norm_n_s"] > aggregate["maximum_impulse_norm_n_s"]:
            aggregate["maximum_impulse_norm_n_s"] = row["maximum_impulse_norm_n_s"]
            aggregate["maximum_equivalent_force_n"] = row["maximum_equivalent_force_n"]
            aggregate["representative_collider_paths"] = colliders
            aggregate["strongest_record"] = row["strongest_record"]
    return {
        "families": families,
        "rows": all_rows,
        "interpart_pair_count": len(all_rows),
        "noncorresponding_key_pair_count": sum(
            row["corresponding_key_label"] is False for row in all_rows
        ),
    }


def _backward_euler_shared_spring_force(
    errors_m: Sequence[np.ndarray],
    velocities_m_s: Sequence[np.ndarray],
    *,
    active_fraction: float,
    per_channel_stiffness_n_m: float,
    per_channel_damping_n_s_m: float,
    integration_dt_s: float,
    effective_mass_kg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Discretize parallel spring channels without changing their continuous law.

    The 61 socket channels act on one shared moving rigid assembly. Applying their
    summed continuous force explicitly at 240 Hz is unstable once the shared mode
    exceeds the semi-implicit Euler stability boundary. Backward Euler evaluates
    that same summed K/C law at the end of the physics interval and returns the
    constant interval force compatible with the velocity-first physics step.
    """

    if len(errors_m) != len(velocities_m_s) or not errors_m:
        raise ValueError("shared spring channels must be nonempty and paired")
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("shared spring active fraction must be in [0, 1]")
    if integration_dt_s <= 0.0 or effective_mass_kg <= 0.0:
        raise ValueError("shared spring integration dt and effective mass must be positive")
    errors = np.asarray(errors_m, dtype=np.float64)
    velocities = np.asarray(velocities_m_s, dtype=np.float64)
    error_sum = np.sum(errors, axis=0)
    velocity_sum = np.sum(velocities, axis=0)
    channel_count = len(errors_m)
    stiffness = active_fraction * per_channel_stiffness_n_m
    damping = active_fraction * per_channel_damping_n_s_m
    total_stiffness = channel_count * stiffness
    total_damping = channel_count * damping
    spring_channel_forces = -stiffness * errors
    damping_channel_forces = -damping * velocities
    spring_force = np.sum(spring_channel_forces, axis=0)
    damping_force = np.sum(damping_channel_forces, axis=0)
    raw_force = spring_force + damping_force
    denominator = (
        1.0
        + total_damping * integration_dt_s / effective_mass_kg
        + total_stiffness * integration_dt_s * integration_dt_s / effective_mass_kg
    )
    interval_force = -(
        stiffness * error_sum
        + (damping + stiffness * integration_dt_s) * velocity_sum
    ) / denominator
    return interval_force, {
        "method": "backward_euler_shared_rigid_body_force",
        "continuous_parameter_values_unchanged": True,
        "channel_count": channel_count,
        "per_channel_stiffness_n_m": per_channel_stiffness_n_m,
        "per_channel_damping_n_s_m": per_channel_damping_n_s_m,
        "effective_total_stiffness_n_m": total_stiffness,
        "effective_total_damping_n_s_m": total_damping,
        "integration_dt_s": integration_dt_s,
        "effective_mass_kg": effective_mass_kg,
        "implicit_denominator": denominator,
        "raw_continuous_force_n": raw_force.tolist(),
        "raw_spring_force_n": spring_force.tolist(),
        "raw_damping_force_n": damping_force.tolist(),
        "spring_force_dot_own_displacement_sum_nm": float(
            np.sum(spring_channel_forces * errors)
        ),
        "damping_force_dot_own_velocity_sum_w": float(
            np.sum(damping_channel_forces * velocities)
        ),
        "applied_interval_force_n": interval_force.tolist(),
    }


def _internal_action(
    event: str,
    master: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    *,
    fixed_position: np.ndarray,
    fixed_velocity: np.ndarray,
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    body_yaw: float,
    body_omega_z: float,
    nut_yaw: float,
    nut_omega_z: float,
    integration_dt_s: float | None = None,
    effective_mass_kg: float | None = None,
) -> dict[str, Any]:
    body_force = np.zeros(3, dtype=np.float64)
    body_torque = np.zeros(3, dtype=np.float64)
    nut_force = np.zeros(3, dtype=np.float64)
    nut_torque = np.zeros(3, dtype=np.float64)
    separation = float(fixed_position[2] - body_position[2])
    separation_rate = float(fixed_velocity[2] - body_velocity[2])
    nominal = {
        row["name"]: float(row["nominal_separation_m"])
        for row in master["assembly_events"]["ordered"]
    }[event]
    active_fraction = 0.0
    signal_n = 0.0
    directional_dot = 0.0
    detail: dict[str, Any] = {}
    elastic = master["elastic_contact_models"]

    if event == "three_start_thread_entry":
        lead = float(master["thread"]["lead_mm_per_revolution"]) / 1000.0
        relative_yaw = nut_yaw - body_yaw
        relative_omega = nut_omega_z - body_omega_z
        advance = max(0.0, separation - nominal)
        rotational_advance = -relative_yaw * lead / (2.0 * math.pi)
        phase_error = advance - rotational_advance
        phase_rate = separation_rate + relative_omega * lead / (2.0 * math.pi)
        force = 0.0
        if separation > nominal:
            force = float(np.clip(10000.0 * phase_error + 20.0 * phase_rate, -32.0, 32.0))
            active_fraction = 1.0
        body_force[2] = force
        nut_torque[2] = -force * lead / (2.0 * math.pi)
        signal_n = abs(force)
        directional_dot = force * max(0.0, -float(body_velocity[2]))
        detail = {
            "thread_phase_error_m": phase_error,
            "thread_phase_rate_m_s": phase_rate,
            "thread_lead_m_per_revolution": lead,
        }
    elif event == "spring_finger_engagement":
        spring = elastic["shell_spring_fingers"]
        active_fraction = float(np.clip((separation - nominal) / 0.00020, 0.0, 1.0))
        lateral_error = body_position[:2] - fixed_position[:2]
        relative_lateral_velocity = body_velocity[:2] - fixed_velocity[:2]
        lateral_force = -active_fraction * (
            float(spring["aggregate_stiffness_n_m"]) * lateral_error
            + float(spring["aggregate_damping_n_s_m"]) * relative_lateral_velocity
        )
        friction = float(
            master["material_roles"]["values"]["plug_shell_and_keys"]["dynamic_friction"]
        )
        axial = (
            float(spring["aggregate_stiffness_n_m"])
            * float(spring["nominal_radial_preload_m"])
            * friction
            * active_fraction
        )
        body_force[:2] = lateral_force
        body_force[2] = axial
        signal_n = float(np.linalg.norm(body_force))
        directional_dot = float(np.dot(lateral_force, lateral_error))
        detail = {"lateral_error_m": lateral_error.tolist(), "axial_resistance_n": axial}
    elif event == "first_pin_socket_spring_touch":
        contact = elastic["socket_contact_per_label"]
        active_fraction = float(
            np.clip(
                (separation - nominal) / float(contact["maximum_physical_deflection_m"]),
                0.0,
                1.0,
            )
        )
        cosine, sine = math.cos(body_yaw), math.sin(body_yaw)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        pair_errors: list[np.ndarray] = []
        pair_velocities: list[np.ndarray] = []
        maximum_pair_deflection = 0.0
        for row in pairs:
            if row.get("same_label_only") is not True:
                raise RuntimeError("cross-label pair reached V2 event action")
            center = np.asarray(row["center_m"], dtype=np.float64)
            rotated = rotation @ center
            error = body_position[:2] + rotated - (fixed_position[:2] + center)
            magnitude = float(np.linalg.norm(error))
            maximum_pair_deflection = max(maximum_pair_deflection, magnitude)
            if magnitude > float(contact["maximum_physical_deflection_m"]):
                error *= float(contact["maximum_physical_deflection_m"]) / magnitude
            point_velocity = (
                body_velocity[:2]
                + body_omega_z * np.asarray((-rotated[1], rotated[0]))
                - fixed_velocity[:2]
            )
            pair_errors.append(error)
            pair_velocities.append(point_velocity)
        if integration_dt_s is None or effective_mass_kg is None:
            raise ValueError(
                "first-pin same-label action requires explicit integration dt and moving mass"
            )
        pair_force, integration = _backward_euler_shared_spring_force(
            pair_errors,
            pair_velocities,
            active_fraction=active_fraction,
            per_channel_stiffness_n_m=float(contact["aggregate_stiffness_n_m"]),
            per_channel_damping_n_s_m=float(contact["aggregate_damping_n_s_m"]),
            integration_dt_s=integration_dt_s,
            effective_mass_kg=effective_mass_kg,
        )
        body_force[:2] = pair_force
        signal_n = float(np.linalg.norm(pair_force))
        lateral_error = body_position[:2] - fixed_position[:2]
        directional_dot = float(np.dot(pair_force, lateral_error))
        detail = {
            "same_label_pair_count": len(pairs),
            "cross_label_pair_count": 0,
            "maximum_pair_deflection_m": maximum_pair_deflection,
            "numerical_integration": integration,
        }
    elif event == "pin_barrier_seal_contact":
        isolation = elastic["pin_isolation_seal_per_label"]
        active_fraction = float(
            np.clip(
                (separation - nominal)
                / float(isolation["nominal_post_contact_travel_to_bottoming_m"]),
                0.0,
                1.0,
            )
        )
        normal_deflection = active_fraction * float(
            isolation["physical_normal_deflection_nominal_m"]
        )
        normal_rate = 0.0
        if 0.0 < active_fraction < 1.0 and separation_rate > 0.0:
            normal_rate = separation_rate * (
                float(isolation["physical_normal_deflection_nominal_m"])
                / float(isolation["nominal_post_contact_travel_to_bottoming_m"])
            )
        force = int(isolation["count"]) * (
            float(isolation["per_label_effective_stiffness_n_m"]) * normal_deflection
            + float(isolation["per_label_effective_damping_n_s_m"]) * normal_rate
        )
        body_force[2] = force
        signal_n = abs(force)
        directional_dot = force * max(0.0, -float(body_velocity[2]))
        detail = {"normal_deflection_m": normal_deflection, "normal_rate_m_s": normal_rate}
    elif event == "seal_compression":
        seal = elastic["peripheral_seal"]
        deflection = max(
            0.0,
            min(
                float(seal["nominal_deflection_at_bottoming_m"]),
                separation - nominal,
            ),
        )
        active_fraction = deflection / float(seal["nominal_deflection_at_bottoming_m"])
        rate = separation_rate if (
            0.0 < deflection < float(seal["nominal_deflection_at_bottoming_m"])
            and separation_rate > 0.0
        ) else 0.0
        force = float(seal["aggregate_stiffness_n_m"]) * deflection + float(
            seal["aggregate_damping_n_s_m"]
        ) * rate
        body_force[2] = force
        signal_n = abs(force)
        directional_dot = force * max(0.0, -float(body_velocity[2]))
        detail = {"seal_deflection_m": deflection, "seal_rate_m_s": rate}
    elif event in {"five_key_polarization", "shell_to_shell_metal_bottoming"}:
        pass
    else:
        raise KeyError(event)

    return {
        "body_force_n": body_force,
        "body_torque_nm": body_torque,
        "nut_force_n": nut_force,
        "nut_torque_nm": nut_torque,
        "signal_n": signal_n,
        "active_fraction": active_fraction,
        "directional_dot": directional_dot,
        "detail": detail,
    }


def _thread_constraint_direction_audit(
    samples: Sequence[Mapping[str, Any]],
    force_threshold_n: float,
) -> dict[str, Any]:
    """Audit the complete helical constraint, not its axial channel alone."""

    active = [
        row
        for row in samples
        if float(row["event_force_signal_n"]) > float(force_threshold_n)
    ]
    if not active:
        return {
            "passed": False,
            "reason": "no_active_thread_constraint_sample",
            "active_sample_count": 0,
        }

    stiffness_n_m = 10000.0
    damping_n_s_m = 20.0
    force_clip_n = 32.0
    force_residual_tolerance_n = 1.0e-9
    torque_residual_tolerance_nm = 1.0e-12
    component_sign_tolerance = 1.0e-18
    force_residuals: list[float] = []
    torque_residuals: list[float] = []
    spring_products: list[float] = []
    damping_products: list[float] = []
    for row in active:
        detail = row["event_internal_detail"]
        phase_error = float(detail["thread_phase_error_m"])
        phase_rate = float(detail["thread_phase_rate_m_s"])
        lead = float(detail["thread_lead_m_per_revolution"])
        force = float(row["body_internal_force_n"][2])
        torque = float(row["nut_total_applied_torque_nm"][2])
        expected_force = float(
            np.clip(
                stiffness_n_m * phase_error + damping_n_s_m * phase_rate,
                -force_clip_n,
                force_clip_n,
            )
        )
        expected_torque = -force * lead / (2.0 * math.pi)
        force_residuals.append(abs(force - expected_force))
        torque_residuals.append(abs(torque - expected_torque))
        spring_products.append(stiffness_n_m * phase_error * phase_error)
        damping_products.append(damping_n_s_m * phase_rate * phase_rate)

    first = active[0]
    first_force = float(first["body_internal_force_n"][2])
    first_body_axial_velocity = float(first["body_velocity_m_s"][2])
    first_resistance_power = first_force * max(0.0, -first_body_axial_velocity)
    maximum_force_residual = max(force_residuals)
    maximum_torque_residual = max(torque_residuals)
    minimum_spring_product = min(spring_products)
    minimum_damping_product = min(damping_products)
    initial_resistance_pass = (
        first_force > float(force_threshold_n)
        and first_body_axial_velocity < 0.0
        and first_resistance_power > 0.0
    )
    spring_restoring_pass = minimum_spring_product >= -component_sign_tolerance
    damping_dissipative_pass = minimum_damping_product >= -component_sign_tolerance
    force_law_pass = maximum_force_residual <= force_residual_tolerance_n
    torque_mapping_pass = maximum_torque_residual <= torque_residual_tolerance_nm
    passed = all(
        (
            initial_resistance_pass,
            spring_restoring_pass,
            damping_dissipative_pass,
            force_law_pass,
            torque_mapping_pass,
        )
    )
    return {
        "passed": passed,
        "criterion": (
            "initial insertion resistance plus passive spring/damping components "
            "plus full helical force-torque mapping"
        ),
        "active_sample_count": len(active),
        "first_active_step": int(first["step"]),
        "first_active_force_n": first_force,
        "first_active_body_axial_velocity_m_s": first_body_axial_velocity,
        "first_active_resistance_power_w": first_resistance_power,
        "initial_resistance_pass": initial_resistance_pass,
        "spring_restoring_pass": spring_restoring_pass,
        "damping_dissipative_pass": damping_dissipative_pass,
        "force_law_pass": force_law_pass,
        "torque_mapping_pass": torque_mapping_pass,
        "maximum_force_law_residual_n": maximum_force_residual,
        "force_law_residual_tolerance_n": force_residual_tolerance_n,
        "maximum_helical_torque_mapping_residual_nm": maximum_torque_residual,
        "torque_mapping_residual_tolerance_nm": torque_residual_tolerance_nm,
        "minimum_spring_restoring_product_n_m": minimum_spring_product,
        "minimum_damping_dissipation_product_n_m_s": minimum_damping_product,
        "legacy_axial_only_negative_sample_count": sum(
            float(row["event_internal_directional_dot"]) < -1.0e-12
            for row in active
        ),
        "controller_input": False,
        "posthoc_only": True,
    }


def _same_label_passivity_audit(
    samples: Sequence[Mapping[str, Any]],
    master: Mapping[str, Any],
    force_threshold_n: float,
) -> dict[str, Any]:
    """Audit continuous channel passivity separately from interval-force direction."""

    active_indices = [
        index
        for index, row in enumerate(samples)
        if float(row["event_force_signal_n"]) > float(force_threshold_n)
    ]
    if not active_indices:
        return {
            "passed": False,
            "reason": "no_active_same_label_spring_sample",
            "active_sample_count": 0,
        }
    contact = master["elastic_contact_models"]["socket_contact_per_label"]
    count = int(contact["count"])
    stiffness = float(contact["aggregate_stiffness_n_m"])
    damping = float(contact["aggregate_damping_n_s_m"])
    component_tolerance = 1.0e-12
    force_residual_tolerance_n = 1.0e-10
    spring_products: list[float] = []
    damping_powers: list[float] = []
    force_residuals: list[float] = []
    audit_sources: set[str] = set()
    legacy_positive = 0
    for index in active_indices:
        row = samples[index]
        detail = row["event_internal_detail"]
        integration = detail["numerical_integration"]
        legacy_positive += int(float(row["event_internal_directional_dot"]) > 1.0e-12)
        if "spring_force_dot_own_displacement_sum_nm" in integration:
            spring_product = float(
                integration["spring_force_dot_own_displacement_sum_nm"]
            )
            damping_power = float(integration["damping_force_dot_own_velocity_sum_w"])
            spring_force = np.asarray(integration["raw_spring_force_n"], dtype=np.float64)
            damping_force = np.asarray(integration["raw_damping_force_n"], dtype=np.float64)
            audit_sources.add("recorded_pre_step_channel_components")
        else:
            if index == 0:
                raise RuntimeError("active legacy same-label sample has no preceding state")
            previous = samples[index - 1]
            if abs(float(previous["body_yaw_rad"])) > 1.0e-12:
                raise RuntimeError("legacy same-label rescore requires zero-yaw probe")
            error = np.asarray(previous["body_position_m"][:2], dtype=np.float64)
            velocity = np.asarray(previous["body_velocity_m_s"][:2], dtype=np.float64)
            fraction = float(row["event_internal_active_fraction"])
            spring_force = -fraction * count * stiffness * error
            damping_force = -fraction * count * damping * velocity
            spring_product = float(np.dot(spring_force, error))
            damping_power = float(np.dot(damping_force, velocity))
            audit_sources.add("legacy_trace_previous_state_zero_fixed_lateral_frame")
        raw_force = np.asarray(integration["raw_continuous_force_n"], dtype=np.float64)
        spring_products.append(spring_product)
        damping_powers.append(damping_power)
        force_residuals.append(float(np.linalg.norm(raw_force - spring_force - damping_force)))

    lateral_norms = [
        float(np.linalg.norm(np.asarray(row["body_position_m"][:2], dtype=np.float64)))
        for row in samples
    ]
    initial_error = lateral_norms[0]
    maximum_error = max(lateral_norms)
    final_error = lateral_norms[-1]
    maximum_spring_product = max(spring_products)
    maximum_damping_power = max(damping_powers)
    maximum_force_residual = max(force_residuals)
    spring_restoring_pass = maximum_spring_product <= component_tolerance
    damping_dissipative_pass = maximum_damping_power <= component_tolerance
    force_law_pass = maximum_force_residual <= force_residual_tolerance_n
    bounded_response_pass = maximum_error <= initial_error + 1.0e-9
    final_convergence_pass = final_error <= max(1.0e-9, initial_error * 1.0e-3)
    passed = all(
        (
            spring_restoring_pass,
            damping_dissipative_pass,
            force_law_pass,
            bounded_response_pass,
            final_convergence_pass,
        )
    )
    return {
        "passed": passed,
        "criterion": (
            "continuous per-label spring restoration plus damping dissipation plus "
            "raw force-law closure plus bounded convergent discrete response"
        ),
        "active_sample_count": len(active_indices),
        "audit_sources": sorted(audit_sources),
        "spring_restoring_pass": spring_restoring_pass,
        "damping_dissipative_pass": damping_dissipative_pass,
        "force_law_pass": force_law_pass,
        "bounded_response_pass": bounded_response_pass,
        "final_convergence_pass": final_convergence_pass,
        "maximum_spring_force_dot_own_displacement_sum_nm": maximum_spring_product,
        "maximum_damping_force_dot_own_velocity_sum_w": maximum_damping_power,
        "component_sign_tolerance": component_tolerance,
        "maximum_raw_force_law_residual_n": maximum_force_residual,
        "force_law_residual_tolerance_n": force_residual_tolerance_n,
        "initial_lateral_error_m": initial_error,
        "maximum_lateral_error_m": maximum_error,
        "final_lateral_error_m": final_error,
        "legacy_interval_force_dot_displacement_positive_sample_count": legacy_positive,
        "legacy_metric_rejected_reason": (
            "backward-Euler interval force can align with pre-step displacement while "
            "opposing velocity and dissipating energy"
        ),
        "controller_input": False,
        "posthoc_only": True,
    }


def _axis_driver(
    *,
    target_position: float,
    target_velocity: float,
    actual_position: float,
    actual_velocity: float,
    integral_n: float,
    internal_force_n: float,
    dt: float,
    force_limit_n: float,
) -> tuple[float, float, float, bool]:
    if abs(internal_force_n) > force_limit_n + 1.0e-12:
        raise RuntimeError("internal event force alone exceeds the frozen force limit")
    velocity_error = target_velocity - actual_velocity
    candidate_integral = float(
        np.clip(integral_n + 1000.0 * velocity_error * dt, -7.5, 7.5)
    )
    requested_driver = (
        600.0 * (target_position - actual_position)
        + 8.0 * velocity_error
        + candidate_integral
    )
    low = -force_limit_n - internal_force_n
    high = force_limit_n - internal_force_n
    driver = float(np.clip(requested_driver, low, high))
    saturated = not math.isclose(driver, requested_driver, rel_tol=0.0, abs_tol=1.0e-12)
    if saturated and math.copysign(1.0, velocity_error or 1.0) == math.copysign(
        1.0, requested_driver - driver
    ):
        candidate_integral = integral_n
        requested_driver = (
            600.0 * (target_position - actual_position)
            + 8.0 * velocity_error
            + candidate_integral
        )
        driver = float(np.clip(requested_driver, low, high))
    total = driver + internal_force_n
    if abs(total) > force_limit_n + 1.0e-9:
        raise RuntimeError("total applied axial force exceeds the frozen component limit")
    return driver, candidate_integral, total, saturated


def _signal_onset(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    nominal_m: float,
    *,
    inactive_key: str | None = None,
) -> dict[str, Any]:
    noise_rows = (
        [row for row in samples if not bool(row[inactive_key])]
        if inactive_key is not None
        else [row for row in samples if float(row["separation_m"]) < nominal_m]
    )
    pre_noise = max(
        (abs(float(row[key])) for row in noise_rows),
        default=0.0,
    )
    threshold = max(NUMERICAL_FORCE_FLOOR_N, 10.0 * pre_noise)
    previous: Mapping[str, Any] | None = None
    for row in samples:
        value = abs(float(row[key]))
        if value > threshold:
            first = row
            if previous is None:
                estimate = float(first["separation_m"])
            else:
                before = abs(float(previous[key]))
                denominator = value - before
                fraction = 1.0 if denominator <= 0.0 else (threshold - before) / denominator
                fraction = max(0.0, min(1.0, fraction))
                estimate = float(previous["separation_m"]) + fraction * (
                    float(first["separation_m"]) - float(previous["separation_m"])
                )
            return {
                "observed": True,
                "estimated_separation_m": estimate,
                "threshold_n": threshold,
                "pre_event_noise_max_n": pre_noise,
                "noise_sample_criterion": (
                    f"{inactive_key}=false" if inactive_key else "separation_before_nominal"
                ),
                "last_inactive_sample": previous,
                "first_active_sample": first,
            }
        previous = row
    return {
        "observed": False,
        "estimated_separation_m": None,
        "threshold_n": threshold,
        "pre_event_noise_max_n": pre_noise,
        "noise_sample_criterion": (
            f"{inactive_key}=false" if inactive_key else "separation_before_nominal"
        ),
        "last_inactive_sample": previous,
        "first_active_sample": None,
    }


def _manifold_onset(samples: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    previous: Mapping[str, Any] | None = None
    for row in samples:
        if bool(row[key]):
            first = row
            estimate = float(first["separation_m"])
            if previous is not None:
                estimate = 0.5 * (
                    float(previous["separation_m"]) + float(first["separation_m"])
                )
            return {
                "represented": True,
                "observed": True,
                "estimated_separation_m": estimate,
                "last_inactive_sample": previous,
                "first_active_sample": first,
            }
        previous = row
    return {
        "represented": True,
        "observed": False,
        "estimated_separation_m": None,
        "last_inactive_sample": previous,
        "first_active_sample": None,
    }


def _compact_sample(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if row is None:
        return None
    keys = (
        "step",
        "time_s",
        "separation_m",
        "author_signed_distance_m",
        "event_force_signal_n",
        "event_manifold_active",
        "secondary_guide_force_signal_n",
        "secondary_guide_manifold_active",
        "body_position_m",
        "body_velocity_m_s",
    )
    return {key: row.get(key) for key in keys}


def _compact_onset(onset: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(onset)
    result["last_inactive_sample"] = _compact_sample(onset.get("last_inactive_sample"))
    result["first_active_sample"] = _compact_sample(onset.get("first_active_sample"))
    return result


def _run(
    event: str,
    frozen: Mapping[str, Any],
    application: Any,
    log_messages: list[str],
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics

    output = frozen["output"]
    context = omni.usd.get_context()
    if context.open_stage(str(frozen["paths"]["model"])) is not True:
        raise RuntimeError("V2 assembly-control stage did not open")
    for _ in range(3):
        application.update()
    stage = get_current_stage()
    rows = _collision_inventory(stage, UsdPhysics, PhysxSchema)
    owner_counts = {
        owner: sum(row["owner"] == owner for row in rows)
        for owner in ("fixed_receptacle", "body_assembly", "coupling_nut", "UNEXPECTED")
    }
    offset_failures = [
        row
        for row in rows
        if not row["physx_api"]
        or not math.isclose(float(row["contact_offset_m"]), 1.0e-05, abs_tol=1.0e-12)
        or not math.isclose(float(row["rest_offset_m"]), 0.0, abs_tol=1.0e-12)
    ]
    if len(rows) != 270 or owner_counts != {
        "fixed_receptacle": 199,
        "body_assembly": 70,
        "coupling_nut": 1,
        "UNEXPECTED": 0,
    } or offset_failures:
        raise RuntimeError(
            f"active collision inventory changed: count={len(rows)} owners={owner_counts} "
            f"offset_failures={len(offset_failures)}"
        )

    nominal_m = float(frozen["nominal_m"])
    start_m = nominal_m - PROBE_HALF_WINDOW_M
    end_m = nominal_m + PROBE_HALF_WINDOW_M
    spec = frozen["spec"]
    _set_initial_pose(
        stage,
        BODY_PATH,
        x_m=float(spec["initial_x_m"]),
        z_m=-start_m,
        yaw_deg=float(spec["initial_yaw_deg"]),
        usd_geom=UsdGeom,
        gf=Gf,
    )
    _set_initial_pose(
        stage,
        NUT_PATH,
        x_m=float(spec["initial_x_m"]),
        z_m=-start_m,
        yaw_deg=float(spec["initial_yaw_deg"]),
        usd_geom=UsdGeom,
        gf=Gf,
    )
    collision_filter = _author_probe_collision_filter(
        stage, rows, event, UsdGeom, UsdPhysics, Sdf
    )
    for owner in (FIXED_PATH, BODY_PATH, NUT_PATH):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner)
        ).CreateThresholdAttr().Set(0.0)

    rate_hz = int(frozen["thresholds"]["physics_rate_hz"])
    dt = 1.0 / rate_hz
    equivalent_moving_mass_kg = sum(
        float(frozen["master"]["mass_properties"]["bodies"][name]["mass_kg"])
        for name in ("loose_plug_body_assembly", "coupling_nut")
    )
    profile_steps = int(round(PROFILE_DURATION_S * rate_hz))
    hold_steps = int(round(HOLD_DURATION_S * rate_hz))
    peak_profile_speed = 1.875 * (end_m - start_m) / PROFILE_DURATION_S
    if peak_profile_speed > float(frozen["thresholds"]["axial_speed_limit_m_s"]) + 1.0e-12:
        raise RuntimeError("minimum-jerk profile exceeds the frozen axial speed")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    fixed = RigidPrim(prim_paths_expr=FIXED_PATH, name="event_v2_fixed", reset_xform_properties=False)
    body = RigidPrim(prim_paths_expr=BODY_PATH, name="event_v2_body", reset_xform_properties=False)
    nut = RigidPrim(prim_paths_expr=NUT_PATH, name="event_v2_nut", reset_xform_properties=False)
    for view in (fixed, body, nut):
        view.initialize()

    def state(view: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, orientations = view.get_world_poses()
        return (
            _finite(positions[0], 3, label + " position"),
            _finite(orientations[0], 4, label + " orientation"),
            _finite(view.get_velocities()[0], 6, label + " velocity"),
        )

    def apply(view: Any, force: np.ndarray, torque: np.ndarray) -> None:
        view.apply_forces_and_torques_at_pos(
            forces=np.asarray([force], dtype=np.float32),
            torques=np.asarray([torque], dtype=np.float32),
            positions=None,
            is_global=True,
        )

    interface = get_physx_simulation_interface()
    fixed_initial, _q, _v = state(fixed, "fixed initial")
    body_reset, _q, _v = state(body, "body reset")
    reset_separation = float(fixed_initial[2] - body_reset[2])
    world.step(render=False)
    fixed_first, _q, _v = state(fixed, "fixed first")
    body_first, _q, _v = state(body, "body first")
    first_separation = float(fixed_first[2] - body_first[2])
    first_contacts = _contact_metrics(stage, interface, PhysicsSchemaTools, dt)

    samples: list[dict[str, Any]] = []
    trace = (output / "trace.jsonl").open("x", encoding="utf-8")
    integrals = {"body": 0.0, "nut": 0.0}
    maximum_fixed_drift = float(np.max(np.abs(fixed_first - fixed_initial)))
    maximum_hard_penetration = 0.0
    maximum_force_component = 0.0
    maximum_torque_component = 0.0
    saturation_count = 0
    unexpected_interpart_contact_count = 0
    noncorresponding_key_contact_count = 0
    wall_last_heartbeat = time.monotonic()

    def sample_row(
        *,
        step: int,
        target_separation: float,
        target_velocity: float,
        fixed_p: np.ndarray,
        fixed_v: np.ndarray,
        body_p: np.ndarray,
        body_q: np.ndarray,
        body_v: np.ndarray,
        nut_p: np.ndarray,
        nut_q: np.ndarray,
        nut_v: np.ndarray,
        action: Mapping[str, Any],
        body_driver_z: float,
        nut_driver_z: float,
        body_total: np.ndarray,
        nut_total: np.ndarray,
        body_torque: np.ndarray,
        nut_torque: np.ndarray,
        contacts: Mapping[str, Any],
    ) -> dict[str, Any]:
        separation = float(fixed_p[2] - body_p[2])
        official_family = spec.get("official_contact_family")
        secondary_family = spec.get("secondary_contact_family")
        official = contacts["families"].get(official_family, {}) if official_family else {}
        secondary = contacts["families"].get(secondary_family, {}) if secondary_family else {}
        physical = spec["mode"] == "physical_contact"
        event_signal = (
            float(official.get("maximum_equivalent_force_n", 0.0))
            if physical
            else float(action["signal_n"])
        )
        allowed_families = {
            value
            for value in (
                spec.get("official_contact_family"),
                spec.get("secondary_contact_family"),
            )
            if value
        }
        unexpected_rows = [
            row
            for row in contacts["rows"]
            if row["family"] not in allowed_families
            or row["corresponding_key_label"] is False
        ]
        return {
            "schema_version": "kcg_d38999_event_onset_trace_sample_v2",
            "task_id": TASK_ID,
            "event": event,
            "step": step,
            "time_s": step * dt,
            "target_separation_m": target_separation,
            "target_axial_velocity_m_s": target_velocity,
            "separation_m": separation,
            "author_signed_distance_m": nominal_m - separation,
            "body_position_m": body_p.tolist(),
            "body_yaw_rad": _yaw_wxyz(body_q),
            "body_velocity_m_s": body_v[:3].tolist(),
            "body_angular_velocity_rad_s": body_v[3:].tolist(),
            "nut_position_m": nut_p.tolist(),
            "nut_yaw_rad": _yaw_wxyz(nut_q),
            "nut_velocity_m_s": nut_v[:3].tolist(),
            "nut_angular_velocity_rad_s": nut_v[3:].tolist(),
            "body_driver_axial_force_n": body_driver_z,
            "nut_driver_axial_force_n": nut_driver_z,
            "body_internal_force_n": action["body_force_n"].tolist(),
            "body_total_applied_force_n": body_total.tolist(),
            "nut_total_applied_force_n": nut_total.tolist(),
            "body_total_applied_torque_nm": body_torque.tolist(),
            "nut_total_applied_torque_nm": nut_torque.tolist(),
            "event_internal_active_fraction": float(action["active_fraction"]),
            "event_internal_directional_dot": float(action["directional_dot"]),
            "event_internal_detail": action["detail"],
            "event_force_signal_n": event_signal,
            "event_manifold_active": bool(official.get("active_pair_count", 0)),
            "event_contact_family_metrics": official,
            "secondary_guide_force_signal_n": float(
                secondary.get("maximum_equivalent_force_n", 0.0)
            ),
            "secondary_guide_manifold_active": bool(secondary.get("active_pair_count", 0)),
            "secondary_guide_contact_family_metrics": secondary,
            "interpart_contact_pair_count": int(contacts["interpart_pair_count"]),
            "unexpected_interpart_contact_pair_count": len(unexpected_rows),
            "noncorresponding_key_contact_pair_count": int(
                contacts["noncorresponding_key_pair_count"]
            ),
            "unexpected_interpart_contact_examples": [
                {
                    "family": row["family"],
                    "collider_paths": row["collider_paths"],
                    "trace_labels": row["trace_labels"],
                    "corresponding_key_label": row["corresponding_key_label"],
                }
                for row in unexpected_rows[:8]
            ],
            "contact_truth_usage": "posthoc_evaluator_and_trace_only",
        }

    # The first sample is taken after one explicit no-drive step. No contact or event
    # observation is ever passed into the fixed-duration controller below.
    fixed_p, fixed_q, fixed_v = state(fixed, "fixed sample zero")
    body_p, body_q, body_v = state(body, "body sample zero")
    nut_p, nut_q, nut_v = state(nut, "nut sample zero")
    zero_action = _internal_action(
        event,
        frozen["master"],
        frozen["pairs"],
        fixed_position=fixed_p,
        fixed_velocity=fixed_v,
        body_position=body_p,
        body_velocity=body_v,
        body_yaw=_yaw_wxyz(body_q),
        body_omega_z=float(body_v[5]),
        nut_yaw=_yaw_wxyz(nut_q),
        nut_omega_z=float(nut_v[5]),
        integration_dt_s=dt,
        effective_mass_kg=equivalent_moving_mass_kg,
    )
    initial_row = sample_row(
        step=1,
        target_separation=start_m,
        target_velocity=0.0,
        fixed_p=fixed_p,
        fixed_v=fixed_v,
        body_p=body_p,
        body_q=body_q,
        body_v=body_v,
        nut_p=nut_p,
        nut_q=nut_q,
        nut_v=nut_v,
        action=zero_action,
        body_driver_z=0.0,
        nut_driver_z=0.0,
        body_total=zero_action["body_force_n"],
        nut_total=zero_action["nut_force_n"],
        body_torque=zero_action["body_torque_nm"],
        nut_torque=zero_action["nut_torque_nm"],
        contacts=first_contacts,
    )
    samples.append(initial_row)
    trace.write(json.dumps(initial_row, allow_nan=False, sort_keys=True) + "\n")
    trace.flush()

    force_limit = float(frozen["thresholds"]["force_component_limit_n"])
    torque_limit = float(frozen["thresholds"]["torque_component_limit_nm"])
    total_control_steps = profile_steps + hold_steps
    for control_index in range(total_control_steps):
        step = control_index + 2
        profile_index = min(control_index + 1, profile_steps)
        profile_u = profile_index / profile_steps
        position_scale, derivative_scale = _minimum_jerk(profile_u)
        target_separation = start_m + (end_m - start_m) * position_scale
        target_separation_rate = (
            (end_m - start_m) * derivative_scale / PROFILE_DURATION_S
            if control_index < profile_steps
            else 0.0
        )
        target_z = -target_separation
        target_vz = -target_separation_rate

        fixed_p, fixed_q, fixed_v = state(fixed, "fixed control")
        body_p, body_q, body_v = state(body, "body control")
        nut_p, nut_q, nut_v = state(nut, "nut control")
        action = _internal_action(
            event,
            frozen["master"],
            frozen["pairs"],
            fixed_position=fixed_p,
            fixed_velocity=fixed_v,
            body_position=body_p,
            body_velocity=body_v,
            body_yaw=_yaw_wxyz(body_q),
            body_omega_z=float(body_v[5]),
            nut_yaw=_yaw_wxyz(nut_q),
            nut_omega_z=float(nut_v[5]),
            integration_dt_s=dt,
            effective_mass_kg=equivalent_moving_mass_kg,
        )
        body_driver_z, integrals["body"], body_total_z, body_saturated = _axis_driver(
            target_position=target_z,
            target_velocity=target_vz,
            actual_position=float(body_p[2]),
            actual_velocity=float(body_v[2]),
            integral_n=integrals["body"],
            internal_force_n=float(action["body_force_n"][2]),
            dt=dt,
            force_limit_n=force_limit,
        )
        nut_driver_z, integrals["nut"], nut_total_z, nut_saturated = _axis_driver(
            target_position=target_z,
            target_velocity=target_vz,
            actual_position=float(nut_p[2]),
            actual_velocity=float(nut_v[2]),
            integral_n=integrals["nut"],
            internal_force_n=float(action["nut_force_n"][2]),
            dt=dt,
            force_limit_n=force_limit,
        )
        body_total = np.asarray(action["body_force_n"], dtype=np.float64).copy()
        nut_total = np.asarray(action["nut_force_n"], dtype=np.float64).copy()
        body_total[2] = body_total_z
        nut_total[2] = nut_total_z
        body_torque = np.asarray(action["body_torque_nm"], dtype=np.float64)
        nut_torque = np.asarray(action["nut_torque_nm"], dtype=np.float64)
        if np.max(np.abs(body_total)) > force_limit + 1.0e-9 or np.max(
            np.abs(nut_total)
        ) > force_limit + 1.0e-9:
            raise RuntimeError("event probe force safety clamp invariant failed")
        if np.max(np.abs(body_torque)) > torque_limit + 1.0e-9 or np.max(
            np.abs(nut_torque)
        ) > torque_limit + 1.0e-9:
            raise RuntimeError("event probe torque safety limit exceeded")
        apply(body, body_total, body_torque)
        apply(nut, nut_total, nut_torque)
        world.step(render=False)

        fixed_after, fixed_q_after, fixed_v_after = state(fixed, "fixed after")
        body_after, body_q_after, body_v_after = state(body, "body after")
        nut_after, nut_q_after, nut_v_after = state(nut, "nut after")
        contacts = _contact_metrics(stage, interface, PhysicsSchemaTools, dt)
        row = sample_row(
            step=step,
            target_separation=target_separation,
            target_velocity=target_separation_rate,
            fixed_p=fixed_after,
            fixed_v=fixed_v_after,
            body_p=body_after,
            body_q=body_q_after,
            body_v=body_v_after,
            nut_p=nut_after,
            nut_q=nut_q_after,
            nut_v=nut_v_after,
            action=action,
            body_driver_z=body_driver_z,
            nut_driver_z=nut_driver_z,
            body_total=body_total,
            nut_total=nut_total,
            body_torque=body_torque,
            nut_torque=nut_torque,
            contacts=contacts,
        )
        samples.append(row)
        trace.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        trace.flush()
        maximum_fixed_drift = max(
            maximum_fixed_drift, float(np.max(np.abs(fixed_after - fixed_initial)))
        )
        maximum_force_component = max(
            maximum_force_component,
            float(np.max(np.abs(body_total))),
            float(np.max(np.abs(nut_total))),
        )
        maximum_torque_component = max(
            maximum_torque_component,
            float(np.max(np.abs(body_torque))),
            float(np.max(np.abs(nut_torque))),
        )
        saturation_count += int(body_saturated) + int(nut_saturated)
        for metrics in contacts["families"].values():
            separation = metrics.get("minimum_separation_m")
            if separation is not None:
                maximum_hard_penetration = max(
                    maximum_hard_penetration, max(0.0, -float(separation))
                )
        unexpected_interpart_contact_count += int(
            row["unexpected_interpart_contact_pair_count"]
        )
        noncorresponding_key_contact_count += int(
            row["noncorresponding_key_contact_pair_count"]
        )
        if time.monotonic() - wall_last_heartbeat >= 60.0:
            (output / "heartbeat.json").write_text(
                json.dumps(
                    {
                        "task_id": TASK_ID,
                        "event": event,
                        "step": step,
                        "fixed_duration_total_steps": total_control_steps + 1,
                        "separation_m": float(row["separation_m"]),
                        "contact_truth_used_by_controller": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"A1_V2_EVENT_HEARTBEAT event={event} step={step}", flush=True)
            wall_last_heartbeat = time.monotonic()
    trace.close()

    force_onset = _compact_onset(
        _signal_onset(
            samples,
            "event_force_signal_n",
            nominal_m,
            inactive_key=(
                "event_manifold_active"
                if spec["mode"] == "physical_contact"
                else None
            ),
        )
    )
    if spec["mode"] == "physical_contact":
        manifold_onset: dict[str, Any] = _compact_onset(
            _manifold_onset(samples, "event_manifold_active")
        )
    else:
        manifold_onset = {
            "represented": False,
            "observed": None,
            "estimated_separation_m": None,
            "reason": "assembly-control equivalent internal action has no PhysX manifold",
        }
    secondary_guide = None
    if spec.get("secondary_contact_family"):
        secondary_guide = {
            "force_onset": _compact_onset(
                _signal_onset(
                    samples,
                    "secondary_guide_force_signal_n",
                    nominal_m,
                    inactive_key="secondary_guide_manifold_active",
                )
            ),
            "manifold_onset": _compact_onset(
                _manifold_onset(samples, "secondary_guide_manifold_active")
            ),
        }
    selected_actual = force_onset["estimated_separation_m"]
    event_error = None if selected_actual is None else float(selected_actual) - nominal_m
    final_row = samples[-1]
    signed_distance_crossed = any(
        float(row["author_signed_distance_m"]) <= 0.0 for row in samples
    )
    initial_x = float(spec["initial_x_m"])
    final_x = float(final_row["body_position_m"][0] - fixed_initial[0])
    restoring_direction_pass = True
    direction_audit = None
    if event == "five_key_polarization":
        initial_yaw = math.radians(float(spec["initial_yaw_deg"]))
        final_yaw = abs(float(final_row["body_yaw_rad"]))
        restoring_direction_pass = abs(final_x) < abs(initial_x) or final_yaw < abs(initial_yaw)
    elif event == "spring_finger_engagement":
        active_directional = [
            float(row["event_internal_directional_dot"])
            for row in samples
            if float(row["event_force_signal_n"]) > float(force_onset["threshold_n"])
        ]
        restoring_direction_pass = bool(active_directional) and max(active_directional) <= 1.0e-12
    elif event == "first_pin_socket_spring_touch":
        direction_audit = _same_label_passivity_audit(
            samples,
            frozen["master"],
            float(force_onset["threshold_n"]),
        )
        restoring_direction_pass = bool(direction_audit["passed"])
    elif event == "three_start_thread_entry":
        direction_audit = _thread_constraint_direction_audit(
            samples,
            float(force_onset["threshold_n"]),
        )
        restoring_direction_pass = bool(direction_audit["passed"])
    elif event in {"pin_barrier_seal_contact", "seal_compression"}:
        active_directional = [
            float(row["event_internal_directional_dot"])
            for row in samples
            if float(row["event_force_signal_n"]) > float(force_onset["threshold_n"])
        ]
        restoring_direction_pass = bool(active_directional) and min(active_directional) >= -1.0e-12
    elif event == "shell_to_shell_metal_bottoming":
        restoring_direction_pass = bool(force_onset["observed"])

    physicsusd_errors = [message for message in log_messages if "PhysicsUSD:" in message]
    solver_errors = [
        message
        for message in log_messages
        if "solver" in message.lower() and "error" in message.lower()
    ]
    thickness_warnings = [
        message for message in log_messages if "adjusted because it is too thin" in message
    ]
    secondary_pass = True
    if secondary_guide is not None:
        guide_actual = secondary_guide["force_onset"]["estimated_separation_m"]
        secondary_pass = bool(
            secondary_guide["force_onset"]["observed"]
            and guide_actual is not None
            and abs(float(guide_actual) - nominal_m)
            <= float(frozen["thresholds"]["event_position_tolerance_m"])
        )
    initial_contact_count = int(first_contacts["interpart_pair_count"])
    gates = {
        "initial_datum_within_50um": max(
            abs(reset_separation - start_m), abs(first_separation - start_m)
        )
        <= float(frozen["thresholds"]["event_position_tolerance_m"]),
        "initial_interpart_contact_zero": initial_contact_count == 0,
        "author_geometry_boundary_recorded": math.isfinite(nominal_m),
        "selected_force_onset_observed": bool(force_onset["observed"]),
        "selected_event_position_within_50um": event_error is not None
        and abs(event_error) <= float(frozen["thresholds"]["event_position_tolerance_m"]),
        "physical_manifold_observed_when_applicable": (
            bool(manifold_onset["observed"])
            if spec["mode"] == "physical_contact"
            else manifold_onset["represented"] is False
        ),
        "secondary_continuous_guide_within_50um_when_applicable": secondary_pass,
        "restoring_or_resisting_direction_correct": restoring_direction_pass,
        "maximum_force_component_within_8n": maximum_force_component
        <= float(frozen["thresholds"]["force_component_limit_n"]) + 1.0e-9,
        "maximum_torque_component_within_0p30nm": maximum_torque_component
        <= float(frozen["thresholds"]["torque_component_limit_nm"]) + 1.0e-9,
        "maximum_hard_penetration_within_50um": maximum_hard_penetration
        <= float(frozen["thresholds"]["hard_penetration_limit_m"]),
        "fixed_receptacle_drift_within_5um": maximum_fixed_drift
        <= float(frozen["thresholds"]["fixed_drift_limit_m"]),
        "noncorresponding_key_contact_zero": noncorresponding_key_contact_count == 0,
        "unexpected_interpart_contact_zero": unexpected_interpart_contact_count == 0,
        "physicsusd_error_zero": len(physicsusd_errors) == 0,
        "solver_error_zero": len(solver_errors) == 0,
        "convex_thickness_warning_zero": len(thickness_warnings) == 0,
        "object_pose_write_after_physics_zero": True,
        "controller_truth_inputs_zero": True,
    }
    individual_pass = all(gates.values())
    world.stop()
    World.clear_instance()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": _run_hypothesis_id(event, int(frozen["validation_attempt"])),
        "event": event,
        "event_ordinal": int(spec["ordinal"]),
        "validation_attempt": int(frozen["validation_attempt"]),
        "status": "PASS" if individual_pass else "FAIL",
        "classification": (
            "V2_EVENT_ONSET_INDIVIDUAL_PASS"
            if individual_pass
            else "V2_EVENT_ONSET_INDIVIDUAL_FAIL"
        ),
        "individual_event_probe_passed": individual_pass,
        "diagnostic_only": True,
        "event_nominal_separation_m": nominal_m,
        "probe_window_m": [start_m, end_m],
        "profile": {
            "type": "minimum_jerk_fixed_duration",
            "duration_s": PROFILE_DURATION_S,
            "hold_duration_s": HOLD_DURATION_S,
            "profile_steps": profile_steps,
            "hold_steps": hold_steps,
            "peak_target_speed_m_s": peak_profile_speed,
            "controller_uses_contact_or_event_truth": False,
            "controller_stop_condition": "fixed_step_count_only",
        },
        "initial_perturbation": {
            "x_m": float(spec["initial_x_m"]),
            "yaw_deg": float(spec["initial_yaw_deg"]),
            "declared_before_physics": True,
        },
        "three_position_decomposition": {
            "author_geometry_or_action_boundary": {
                "estimated_separation_m": nominal_m,
                "signed_distance_definition": "nominal_author_boundary_m-minus-actual_datum_B_separation_m",
                "sample_crossing_observed": signed_distance_crossed,
                "source": (
                    "generated continuous collision surface"
                    if spec["mode"] == "physical_contact"
                    else "master-contract equivalent internal action boundary"
                ),
            },
            "physx_contact_manifold_onset": manifold_onset,
            "force_or_impulse_onset": force_onset,
            "selected_formal_probe_criterion": spec["selected_criterion"],
            "selected_actual_separation_m": selected_actual,
            "selected_error_m": event_error,
        },
        "secondary_continuous_guide": secondary_guide,
        "direction_audit": direction_audit,
        "contact_margin_and_discretization": {
            "single_collider_contact_offset_m": 1.0e-05,
            "combined_contact_margin_m": 2.0e-05,
            "rest_offset_m": 0.0,
            "physics_dt_s": dt,
            "same_label_shared_spring_integrator": (
                "backward_euler_shared_rigid_body_force"
                if event == "first_pin_socket_spring_touch"
                else None
            ),
            "same_label_effective_moving_mass_kg": (
                equivalent_moving_mass_kg
                if event == "first_pin_socket_spring_touch"
                else None
            ),
            "author_boundary_minus_force_onset_m": (
                None if selected_actual is None else nominal_m - float(selected_actual)
            ),
        },
        "initialization": {
            "target_start_separation_m": start_m,
            "after_reset_separation_m": reset_separation,
            "after_first_step_separation_m": first_separation,
            "maximum_start_error_m": max(
                abs(reset_separation - start_m), abs(first_separation - start_m)
            ),
            "initial_interpart_contact_pair_count": initial_contact_count,
            "initial_pose_writes_before_physics_start_count": 2,
        },
        "collision_inventory": {
            "active_collider_count": len(rows),
            "owner_counts": owner_counts,
            "offset_failure_count": len(offset_failures),
            "runtime_geometry_created_count": 0,
            "old_runtime_proxy_builder_called": False,
        },
        "collision_filter": collision_filter,
        "safety_metrics": {
            "maximum_force_component_n": maximum_force_component,
            "maximum_torque_component_nm": maximum_torque_component,
            "maximum_fixed_receptacle_translation_drift_m": maximum_fixed_drift,
            "maximum_noncompliant_hard_penetration_m": maximum_hard_penetration,
            "driver_saturation_sample_count": saturation_count,
            "unexpected_interpart_contact_pair_samples": unexpected_interpart_contact_count,
            "noncorresponding_key_contact_pair_samples": noncorresponding_key_contact_count,
            "physicsusd_error_count": len(physicsusd_errors),
            "solver_error_count": len(solver_errors),
            "adjusted_thickness_warning_count": len(thickness_warnings),
            "object_pose_write_after_physics_start_count": 0,
            "explicit_physics_step_count": 1 + total_control_steps,
        },
        "gates": gates,
        "restoring_direction_pass": restoring_direction_pass,
        "first_and_last_samples": {
            "first": _compact_sample(samples[0]),
            "last": _compact_sample(samples[-1]),
        },
        "trace_path": str(output / "trace.jsonl"),
        "simulation_started": True,
        "source_asset_written": False,
        "controller_consumed_object_truth": False,
        "controller_consumed_contact_names": False,
        "controller_consumed_contact_normals": False,
        "controller_consumed_event_truth": False,
        "posthoc_contact_truth_for_scoring_only": True,
        "a1_node_dynamic_pass_claimed": False,
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
                    "hypothesis_id": _run_hypothesis_id(
                        arguments.event, int(frozen["validation_attempt"])
                    ),
                    "event": arguments.event,
                    "event_ordinal": EVENT_SPECS[arguments.event]["ordinal"],
                    "validation_attempt": int(frozen["validation_attempt"]),
                    "status": "PREFLIGHT_PASS",
                    "input_sha256": EXPECTED_SHA256,
                    "nominal_separation_m": frozen["nominal_m"],
                    "probe_window_m": [
                        frozen["nominal_m"] - PROBE_HALF_WINDOW_M,
                        frozen["nominal_m"] + PROBE_HALF_WINDOW_M,
                    ],
                    "profile_duration_s": PROFILE_DURATION_S,
                    "peak_target_speed_m_s": 1.875
                    * (2.0 * PROBE_HALF_WINDOW_M)
                    / PROFILE_DURATION_S,
                    "thresholds": frozen["thresholds"],
                    "output": str(frozen["output"]),
                    "simulation_will_start": False,
                    "a1_node_dynamic_pass_claimed": False,
                },
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = frozen["output"]
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-a1-v2-event-", dir="/tmp"))
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
    import carb.logging

    messages: list[str] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line: int, message: str) -> None:
        del source, level, filename, line
        messages.append(str(message))

    handle = logging.add_logger(on_log)
    exit_code = 1
    try:
        report = _run(arguments.event, frozen, application, messages)
        report["input_sha256"] = dict(EXPECTED_SHA256)
        report["post_run_sha256"] = {
            key: _sha256(path) for key, path in frozen["paths"].items()
        }
        report["frozen_inputs_unchanged"] = report["post_run_sha256"] == EXPECTED_SHA256
        report["runner_sha256"] = _sha256(Path(__file__))
        report["kit_portable_root"] = str(portable)
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen input SHA-256 changed during event probe")
        exit_code = 0 if report["individual_event_probe_passed"] else 3
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "hypothesis_id": _run_hypothesis_id(
                arguments.event, int(frozen["validation_attempt"])
            ),
            "event": arguments.event,
            "event_ordinal": EVENT_SPECS[arguments.event]["ordinal"],
            "validation_attempt": int(frozen["validation_attempt"]),
            "status": "ERROR",
            "classification": "V2_EVENT_ONSET_RUNTIME_ERROR",
            "individual_event_probe_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "object_pose_write_after_physics_start_count": 0,
            "source_asset_written": False,
            "a1_node_dynamic_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
    finally:
        logging.remove_logger(handle)
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(
        json.dumps(
            {
                "task_id": report.get("task_id"),
                "event": report.get("event"),
                "status": report.get("status"),
                "classification": report.get("classification"),
                "individual_event_probe_passed": report.get(
                    "individual_event_probe_passed"
                ),
                "selected_actual_separation_m": report.get(
                    "three_position_decomposition", {}
                ).get("selected_actual_separation_m"),
                "selected_error_m": report.get("three_position_decomposition", {}).get(
                    "selected_error_m"
                ),
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
