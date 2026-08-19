#!/usr/bin/env python3

"""Generate the three D38999 multilayer representations from one frozen contract.

This program is deliberately usable with the repository Python interpreter.  It
authors deterministic USDA text and machine-readable mapping evidence without
starting Isaac Sim.  Runtime acceptance remains a separate task.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_RELATIVE_PATH = Path(
    "src/kcg_connector/isaac/build_d38999_multilayer_models.py"
)
CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
)
PHYSICAL_CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_assembly_control_authorized_overrides_v2.yaml"
)
OUTPUT_ROOT_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1"
)
RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/BUILD_RESULT.json"
)
EVENT_ONSET_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/DYN-A1-EVENT-ONSET-CALIBRATION/BUILD_RESULT.json"
)
EVENT_ONSET_BACKUP_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/DYN-A1-EVENT-ONSET-CALIBRATION/"
    "D38999_ASSEMBLY_CONTROL_V1_BEFORE.usda"
)
NUT_GRASP_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/DYN-B2-B4-THREE-FINGER-GRASP/BUILD_RESULT.json"
)
NUT_GRASP_BACKUP_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/DYN-B2-B4-THREE-FINGER-GRASP/"
    "D38999_ASSEMBLY_CONTROL_V1_BEFORE_GRASP.usda"
)
V2_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/ASSET_BUILD_RESULT.json"
)
V2_BACKUP_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/"
    "PRE_V2_ASSEMBLY_CONTROL_9630B31F.usda"
)
V2_FINE_OFFSET_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/FINE_OFFSET_FIX_RESULT.json"
)
V2_FINE_OFFSET_BACKUP_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/"
    "PRE_FINE_OFFSET_ASSEMBLY_D2E27ACB.usda"
)
V2_SHOULDER_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A2-NOMINAL-INSERTION-V2/"
    "A2_RUN05_NUT_BODY_SHOULDER_TARGETED_FIX_RESULT.json"
)
V2_SHOULDER_BACKUP_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A2-NOMINAL-INSERTION-V2/"
    "PRE_H6_ASSEMBLY_CONTROL_D5BCC5E8.usda"
)
V2_SHOULDER_ROOTCAUSE_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A2-NOMINAL-INSERTION-V2/"
    "A2_RUN05_NUT_BODY_SHOULDER_ROOTCAUSE.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
)
EXPECTED_PHYSICAL_CONTRACT_SHA256 = (
    "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
)
REPRESENTATIONS = (
    "D38999_VISUAL_COMPLETE_V1",
    "D38999_ASSEMBLY_CONTROL_V1",
    "D38999_LOCAL_CONTACT_REFERENCE_V1",
)
HIGH_DETAIL_ROOT_PRIM = "/World/D38999Shell25JKeyedPhysicalV3"
MULTILAYER_PARENT_PRIM = "/World/D38999MultilayerV1"
PAIR_NAME = "D38999Pair"
RING_SEGMENTS = 64
LOCAL_MILLIMETRE_TO_METRE = 0.001
CONVEX_MIN_THICKNESS_LOCAL_MM = 0.001
EXPECTED_PRE_ONSET_ASSEMBLY_SHA256 = (
    "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec"
)
EXPECTED_PRE_GRASP_ASSEMBLY_SHA256 = (
    "f0b07519f107f8cc35bca3230d1fd7352ea529c4ab77f7e969771cee75687874"
)
EXPECTED_PRE_V2_ASSEMBLY_SHA256 = (
    "9630b31f7fcbf50352038e2949ac2fcb2ba163ff143248e50f86c7d5bf2da040"
)
EXPECTED_PRE_FINE_OFFSET_ASSEMBLY_SHA256 = (
    "d2e27acb3ccb8de6cf4ffad3d40940e3f8bcf2a4ba30dc223a8f2be11fdf9ea0"
)
EXPECTED_PRE_FINE_OFFSET_GENERATOR_SHA256 = (
    "8eec5cacc80e5dd7eff82af5b24735707eaefc7d2517a8e0f40e8789c5e74984"
)
EXPECTED_PRE_FINE_OFFSET_RESULT_SHA256 = (
    "2a873c10db375b7d6eb61606dc6d3315772470ad0afe55a2b834c2cebac955fa"
)
EXPECTED_PRE_SHOULDER_ASSEMBLY_SHA256 = (
    "d5bcc5e8b28e31912f65cd87a0bbe5d7a035744f7f7d8c7b785e17cdad382a6e"
)
EXPECTED_SHOULDER_ROOTCAUSE_SHA256 = (
    "5f491d90c67d49d856120e8d6aca6a2abab44714bee05bc714f86bc188470438"
)
FINE_CONNECTOR_CONTACT_OFFSET_M = 1.0e-05
FINE_CONNECTOR_REST_OFFSET_M = 0.0
EXPECTED_ACTIVE_FINE_COLLIDER_COUNT = 270
EXPECTED_POST_SHOULDER_ACTIVE_FINE_COLLIDER_COUNT = 278
AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M = (0.009, 0.029)
AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M = 0.024
EXPECTED_PRESERVED_OUTPUT_SHA256 = {
    "D38999_VISUAL_COMPLETE_V1.usda": (
        "69fe6dc3ca9caace8bb26cd0cfad68c0eb84111f09697da6068cd91802d65c0a"
    ),
    "D38999_LOCAL_CONTACT_REFERENCE_V1.usda": (
        "94b9cea0a7bb1e4d4a7c6583819abe1c722e252ae45012ba78f1a396b0a5ab85"
    ),
    "MODEL_MAPPING.json": (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all three D38999 multilayer representations"
    )
    parser.add_argument(
        "--contract",
        default=str(WORKSPACE_ROOT / CONTRACT_RELATIVE_PATH),
    )
    parser.add_argument(
        "--output-root",
        default=str(WORKSPACE_ROOT / OUTPUT_ROOT_RELATIVE_PATH),
    )
    parser.add_argument(
        "--result",
        default=str(WORKSPACE_ROOT / RESULT_RELATIVE_PATH),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all frozen inputs and print the write plan without writing",
    )
    parser.add_argument(
        "--event-onset-calibration",
        action="store_true",
        help=(
            "Guarded DYN-A1 update of D38999_ASSEMBLY_CONTROL_V1 only; "
            "visual, local-reference, and historical mapping files stay byte-identical"
        ),
    )
    parser.add_argument(
        "--initial-nut-grasp-collision",
        action="store_true",
        help=(
            "Guarded DYN-B2-B4 update of D38999_ASSEMBLY_CONTROL_V1 only; "
            "adds the human-authorized coupling-nut rear grasp collider"
        ),
    )
    parser.add_argument(
        "--authorized-overrides-v2",
        action="store_true",
        help=(
            "Guarded V2 composition from the master contract plus the one "
            "authorized override file; writes only D38999_ASSEMBLY_CONTROL_V1"
        ),
    )
    parser.add_argument(
        "--v2-fine-collision-offset-fix",
        action="store_true",
        help=(
            "Guarded second-stage V2 update that adds only the physical "
            "contract's explicit fine-connector contact/rest offsets"
        ),
    )
    parser.add_argument(
        "--v2-nut-body-shoulder-repair",
        action="store_true",
        help=(
            "Guarded A2 H6 repair that restores only the frozen physical "
            "nut-to-body shoulder load path in D38999_ASSEMBLY_CONTROL_V1"
        ),
    )
    parser.add_argument(
        "--authorized-overrides",
        default=str(WORKSPACE_ROOT / AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()


def _require_exact_path(requested: str, expected_relative: Path, role: str) -> Path:
    requested_path = Path(requested).expanduser().resolve()
    expected_path = (WORKSPACE_ROOT / expected_relative).resolve()
    if requested_path != expected_path:
        raise PermissionError(
            f"{role} path is frozen: requested={requested_path}, expected={expected_path}"
        )
    return expected_path


def _load_and_validate_authorized_overrides_v2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("V2 authorized overrides must be a mapping")
    required_exact: dict[str, Any] = {
        "schema_version": "kcg_d38999_assembly_control_authorized_overrides_v2",
        "task_id": "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2",
        "status": "AUTHORIZED",
        "event_onset_proxy_version": 2,
        "initial_nut_grasp_collision_enabled": True,
        "initial_nut_grasp_local_z_interval_m": [0.009, 0.029],
        "initial_nut_grasp_outer_radius_m": 0.024,
        "new_backshell_enabled": False,
        "visual_model_change_allowed": False,
        "local_reference_change_allowed": False,
        "model_mapping_change_allowed": False,
        "master_contract_change_allowed": False,
        "post_physics_pose_write_allowed": False,
        "controller_truth_input_allowed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    mismatches = {
        key: {"actual": document.get(key), "expected": expected}
        for key, expected in required_exact.items()
        if document.get(key) != expected
    }
    expected_contract = {
        "path": CONTRACT_RELATIVE_PATH.as_posix(),
        "sha256": EXPECTED_CONTRACT_SHA256,
    }
    if document.get("source_contract") != expected_contract:
        mismatches["source_contract"] = {
            "actual": document.get("source_contract"),
            "expected": expected_contract,
        }
    expected_pre_v2 = {
        "path": (OUTPUT_ROOT_RELATIVE_PATH / "D38999_ASSEMBLY_CONTROL_V1.usda").as_posix(),
        "sha256": EXPECTED_PRE_V2_ASSEMBLY_SHA256,
    }
    if document.get("pre_v2_assembly_control") != expected_pre_v2:
        mismatches["pre_v2_assembly_control"] = {
            "actual": document.get("pre_v2_assembly_control"),
            "expected": expected_pre_v2,
        }
    if mismatches:
        raise PermissionError(f"V2 authorized override validation failed: {mismatches}")
    return document


def _load_and_validate_contract(contract_path: Path) -> dict[str, Any]:
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    actual_sha = _sha256(contract_path)
    if actual_sha != EXPECTED_CONTRACT_SHA256:
        raise PermissionError(
            "master contract fingerprint changed: "
            f"actual={actual_sha}, expected={EXPECTED_CONTRACT_SHA256}"
        )
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("master contract must be a mapping")
    required_exact = {
        ("schema_version",): "kcg_d38999_master_model_contract_v1",
        ("status",): "FROZEN_FOR_MULTILAYER_V1",
        ("scope", "shared_generator"): GENERATOR_RELATIVE_PATH.as_posix(),
        ("scope", "simulation_only"): True,
        ("scope", "hardware_authorized"): False,
        ("contact_layout", "pin_socket_pairs"): 61,
        ("contact_layout", "same_label_pairing_required"): True,
        ("contact_layout", "cross_label_effect_allowed"): False,
        ("keying", "key_count"): 5,
        ("keying", "keyway_count"): 5,
        ("assembly_events", "event_count"): 7,
        ("thread", "starts"): 3,
        ("anti_decoupling", "cycle_count_per_revolution"): 36,
        ("truth_firewall", "internal_model_state_exposed_to_task_controller"): False,
        ("authorization", "generate_three_multilayer_representations"): True,
        ("authorization", "modify_high_detail_reference"): False,
        ("authorization", "overwrite_high_detail_reference"): False,
        ("authorization", "create_second_geometry_candidate"): False,
        ("authorization", "increase_socket_diameter"): False,
        ("authorization", "increase_force_limit"): False,
        ("authorization", "relax_formal_threshold"): False,
        ("authorization", "run_task_r12_006d"): False,
        ("authorization", "run_task_r12_006e"): False,
        ("authorization", "enter_robot_grasp_vision_or_rl"): False,
        ("authorization", "claim_formal_r12_complete"): False,
        ("authorization", "hardware_authorized"): False,
    }
    for path, expected in required_exact.items():
        value: Any = document
        for key in path:
            value = value[key]
        if value != expected:
            raise PermissionError(
                f"frozen contract guard failed at {'.'.join(path)}: "
                f"actual={value!r}, expected={expected!r}"
            )
    labels = [str(row["label"]) for row in document["contact_layout"]["pairs"]]
    if len(labels) != 61 or len(set(labels)) != 61:
        raise ValueError("contact labels must contain exactly 61 unique entries")
    if len(document["keying"]["canonical_angles_deg"]) != 5:
        raise ValueError("five key angles are required")
    if len(document["assembly_events"]["ordered"]) != 7:
        raise ValueError("seven assembly events are required")
    return document


def _load_and_validate_physical_shoulder_recipe(
    physical_contract_path: Path, master_document: Mapping[str, Any]
) -> dict[str, Any]:
    if not physical_contract_path.is_file():
        raise FileNotFoundError(physical_contract_path)
    actual_sha = _sha256(physical_contract_path)
    if actual_sha != EXPECTED_PHYSICAL_CONTRACT_SHA256:
        raise PermissionError(
            "physical contract fingerprint changed: "
            f"actual={actual_sha}, expected={EXPECTED_PHYSICAL_CONTRACT_SHA256}"
        )
    document = yaml.safe_load(physical_contract_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("physical contract must be a mapping")
    bearing = document["physical_proxy_boundaries"]["nut_body_bearing"]
    master_bearing = master_document["coupling_nut_motion"]
    geometry = bearing["physical_shoulder_collision_geometry"]
    exact = {
        "representation": (
            bearing["representation"],
            "generic_D6_cylindrical_bearing_proxy_with_physical_shoulders",
        ),
        "joint_backup_low_m": (
            bearing["transZ_joint_backup_limits_m"]["low"], -0.00015
        ),
        "joint_backup_high_m": (
            bearing["transZ_joint_backup_limits_m"]["high"], 0.00015
        ),
        "shoulder_endplay_low_m": (
            bearing["physical_shoulder_contact_endplay_m"]["low"], -0.00005
        ),
        "shoulder_endplay_high_m": (
            bearing["physical_shoulder_contact_endplay_m"]["high"], 0.00005
        ),
        "shoulder_before_backup": (
            bearing["shoulder_contacts_before_joint_backup_limit"], True
        ),
        "load_path": (
            bearing["load_path"],
            "thread_to_nut_to_physical_shoulder_to_plug_body",
        ),
        "geometry_representation": (
            geometry["representation"],
            "two_analytic_axial_caps_each_against_three_analytic_spheres",
        ),
        "body0_cap_radius_m": (geometry["body0_cap_radius_m"], 0.0196),
        "body0_cap_axial_thickness_m": (
            geometry["body0_cap_axial_thickness_m"], 0.0003
        ),
        "body1_sphere_count_per_direction": (
            geometry["body1_sphere_count_per_direction"], 3
        ),
        "body1_sphere_radius_m": (geometry["body1_sphere_radius_m"], 0.0005),
        "body1_sphere_distribution_radius_m": (
            geometry["body1_sphere_distribution_radius_m"], 0.0185
        ),
        "body1_sphere_phases_deg": (
            geometry["body1_sphere_phases_deg"], [0.0, 120.0, 240.0]
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in exact.items()
        if actual != expected
    }
    if master_bearing["transZ_backup_limits_m"] != bearing[
        "transZ_joint_backup_limits_m"
    ]:
        mismatches["master_joint_backup_limits"] = {
            "actual": master_bearing["transZ_backup_limits_m"],
            "expected": bearing["transZ_joint_backup_limits_m"],
        }
    if master_bearing["physical_shoulder_endplay_m"] != bearing[
        "physical_shoulder_contact_endplay_m"
    ]:
        mismatches["master_shoulder_endplay"] = {
            "actual": master_bearing["physical_shoulder_endplay_m"],
            "expected": bearing["physical_shoulder_contact_endplay_m"],
        }
    if mismatches:
        raise PermissionError(f"frozen shoulder recipe guard failed: {mismatches}")
    return bearing


def _validate_authoritative_inputs(document: Mapping[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for key, row in document["authoritative_inputs"].items():
        if key in {"source_class", "source_ref"}:
            continue
        source_path = (WORKSPACE_ROOT / str(row["path"])).resolve()
        try:
            source_path.relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise PermissionError(f"authoritative input escapes workspace: {source_path}") from exc
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual = _sha256(source_path)
        expected = str(row["sha256"])
        if actual != expected:
            raise PermissionError(
                f"authoritative input fingerprint changed for {key}: "
                f"actual={actual}, expected={expected}"
            )
        verified[_repo_relative(source_path)] = actual
    return verified


def _usd_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _usd_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("USDA numeric values must be finite")
    if number == 0.0:
        return "0"
    return f"{number:.12g}"


def _usd_bool(value: Any) -> str:
    return "1" if bool(value) else "0"


def _fine_connector_collision_attribute_lines() -> list[str]:
    return [
        "    float physxCollision:contactOffset = "
        + _usd_number(FINE_CONNECTOR_CONTACT_OFFSET_M),
        "    float physxCollision:restOffset = "
        + _usd_number(FINE_CONNECTOR_REST_OFFSET_M),
    ]


def _usd_string_array(values: Iterable[Any]) -> str:
    return "[" + ", ".join(_usd_string(value) for value in values) + "]"


def _usd_vec(values: Sequence[Any]) -> str:
    return "(" + ", ".join(_usd_number(value) for value in values) + ")"


def _indent(lines: Iterable[str], depth: int = 1) -> list[str]:
    prefix = "    " * depth
    return [prefix + line if line else "" for line in lines]


def _stage_header() -> list[str]:
    return [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    kilogramsPerUnit = 1",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
    ]


def _representation_metadata(
    *, representation: str, contract_sha: str, generator_sha: str
) -> list[str]:
    return [
        f"custom string kcg:representationId = {_usd_string(representation)}",
        f"custom string kcg:masterContractPath = {_usd_string(CONTRACT_RELATIVE_PATH.as_posix())}",
        f"custom string kcg:masterContractSha256 = {_usd_string(contract_sha)}",
        f"custom string kcg:sharedGeneratorPath = {_usd_string(GENERATOR_RELATIVE_PATH.as_posix())}",
        f"custom string kcg:sharedGeneratorSha256 = {_usd_string(generator_sha)}",
        "custom bool kcg:simulationOnly = 1",
        "custom bool kcg:hardwareAuthorized = 0",
        "custom bool kcg:formalR12Frozen = 0",
    ]


def _reference_asset_text(
    *,
    representation: str,
    source_relative_to_output: str,
    source_repo_path: str,
    source_sha: str,
    contract_sha: str,
    generator_sha: str,
    read_only: bool,
    complete_robot_mainline_allowed: bool,
) -> str:
    lines = _stage_header()
    lines += ['def Xform "World"', "{"]
    lines += _indent(['def Xform "D38999MultilayerV1"', "{"])
    lines += _indent([f'def Xform "{representation}"', "{"], 2)
    lines += _indent(
        _representation_metadata(
            representation=representation,
            contract_sha=contract_sha,
            generator_sha=generator_sha,
        ),
        3,
    )
    lines += _indent(
        [
            f"custom string kcg:referenceSourcePath = {_usd_string(source_repo_path)}",
            f"custom string kcg:referenceSourceSha256 = {_usd_string(source_sha)}",
            f"custom bool kcg:readOnlyReference = {_usd_bool(read_only)}",
            "custom bool kcg:visibleGeometryPreserved = 1",
            "custom int kcg:visiblePinCount = 61",
            "custom int kcg:visibleSocketCount = 61",
            "custom bool kcg:visibleThreadPreserved = 1",
            "custom bool kcg:visibleDetentTeethPreserved = 1",
            "custom bool kcg:visibleSealsPreserved = 1",
            "custom bool kcg:everyVisibleDetailDynamicCollisionRequired = 0",
            "custom bool kcg:completeRobotAssemblyMainlineAllowed = "
            + _usd_bool(complete_robot_mainline_allowed),
            "",
            f'def Xform "{PAIR_NAME}" (',
            f"    prepend references = @{source_relative_to_output}@<{HIGH_DETAIL_ROOT_PRIM}>",
            ")",
            "{",
            "}",
        ],
        3,
    )
    lines += _indent(["}"], 2)
    lines += _indent(["}"], 1)
    lines += ["}", ""]
    return "\n".join(lines)


def _ring_mesh_lines(
    *,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z0: float,
    z1: float,
    collision_role: str,
    color: Sequence[float],
    collision_enabled: bool = True,
) -> list[str]:
    if not (0.0 < inner_radius < outer_radius and z0 < z1):
        raise ValueError(f"invalid annular mesh dimensions for {name}")
    points: list[tuple[float, float, float]] = []
    for index in range(RING_SEGMENTS):
        angle = 2.0 * math.pi * index / RING_SEGMENTS
        cosine, sine = math.cos(angle), math.sin(angle)
        points.extend(
            [
                (inner_radius * cosine, inner_radius * sine, z0),
                (outer_radius * cosine, outer_radius * sine, z0),
                (inner_radius * cosine, inner_radius * sine, z1),
                (outer_radius * cosine, outer_radius * sine, z1),
            ]
        )
    counts: list[int] = []
    indices: list[int] = []
    for index in range(RING_SEGMENTS):
        current = 4 * index
        following = 4 * ((index + 1) % RING_SEGMENTS)
        faces = (
            (current, following, following + 1, current + 1),
            (current + 2, current + 3, following + 3, following + 2),
            (current, current + 2, following + 2, following),
            (current + 1, following + 1, following + 3, current + 3),
        )
        for face in faces:
            counts.append(4)
            indices.extend(face)
    point_text = ", ".join(_usd_vec(point) for point in points)
    count_text = ", ".join(str(value) for value in counts)
    index_text = ", ".join(str(value) for value in indices)
    declaration = [f'def Mesh "{name}"']
    if collision_enabled:
        declaration += [
            "(",
            '    apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]',
            ")",
        ]
    return [
        *declaration,
        "{",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
        "    custom string kcg:surfaceClass = \"single_connected_annular_mesh\"",
        f"    custom int kcg:circumferentialSamples = {RING_SEGMENTS}",
        f"    point3f[] points = [{point_text}]",
        f"    int[] faceVertexCounts = [{count_text}]",
        f"    int[] faceVertexIndices = [{index_text}]",
        (
            f"    bool physics:collisionEnabled = {_usd_bool(collision_enabled)}"
            if collision_enabled
            else "    custom bool kcg:dynamicCollisionEnabled = 0"
        ),
        *(_fine_connector_collision_attribute_lines() if collision_enabled else []),
        *(['    uniform token physics:approximation = "none"'] if collision_enabled else []),
        f"    color3f[] primvars:displayColor = [{_usd_vec(color)}]",
        '    uniform token subdivisionScheme = "none"',
        "}",
    ]


def _convex_arc_mesh_lines(
    *,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z0: float,
    z1: float,
    angle0_rad: float,
    angle1_rad: float,
    collision_role: str,
    trace_label: str,
    color: Sequence[float],
) -> list[str]:
    if not (
        0.0 < inner_radius < outer_radius
        and z0 < z1
        and angle0_rad < angle1_rad
        and angle1_rad - angle0_rad <= math.pi / 2.0
    ):
        raise ValueError(f"invalid convex arc dimensions for {name}")
    c0, s0 = math.cos(angle0_rad), math.sin(angle0_rad)
    c1, s1 = math.cos(angle1_rad), math.sin(angle1_rad)
    points_m = (
        (inner_radius * c0, inner_radius * s0, z0),
        (outer_radius * c0, outer_radius * s0, z0),
        (outer_radius * c1, outer_radius * s1, z0),
        (inner_radius * c1, inner_radius * s1, z0),
        (inner_radius * c0, inner_radius * s0, z1),
        (outer_radius * c0, outer_radius * s0, z1),
        (outer_radius * c1, outer_radius * s1, z1),
        (inner_radius * c1, inner_radius * s1, z1),
    )
    points_local_mm = [
        tuple(component / LOCAL_MILLIMETRE_TO_METRE for component in point)
        for point in points_m
    ]
    return [
        f'def Mesh "{name}" (',
        '    apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI", "PhysxCollisionAPI"]',
        ")",
        "{",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
        f"    custom string kcg:traceLabel = {_usd_string(trace_label)}",
        f"    point3f[] points = [{', '.join(_usd_vec(point) for point in points_local_mm)}]",
        "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]",
        "    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]",
        "    bool physics:collisionEnabled = 1",
        *_fine_connector_collision_attribute_lines(),
        '    uniform token physics:approximation = "convexHull"',
        f"    float physxConvexHullCollision:minThickness = {_usd_number(CONVEX_MIN_THICKNESS_LOCAL_MM)}",
        f"    color3f[] primvars:displayColor = [{_usd_vec(color)}]",
        '    uniform token subdivisionScheme = "none"',
        f"    double3 xformOp:scale = {_usd_vec((LOCAL_MILLIMETRE_TO_METRE,) * 3)}",
        '    uniform token[] xformOpOrder = ["xformOp:scale"]',
        "}",
    ]


def _split_arc(angle0: float, angle1: float) -> list[tuple[float, float]]:
    maximum_span = 2.0 * math.pi / RING_SEGMENTS
    count = max(1, int(math.ceil((angle1 - angle0) / maximum_span)))
    return [
        (
            angle0 + (angle1 - angle0) * index / count,
            angle0 + (angle1 - angle0) * (index + 1) / count,
        )
        for index in range(count)
    ]


def _annular_segment_group_lines(
    *,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z0: float,
    z1: float,
    collision_role: str,
    color: Sequence[float],
    intervals_rad: Sequence[tuple[float, float]] = ((0.0, 2.0 * math.pi),),
    segment_prefix: str = "Segment",
) -> list[str]:
    lines = [
        f'def Xform "{name}"',
        "{",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
        '    custom string kcg:surfaceClass = "exact_convex_annular_segments"',
        f"    custom double kcg:worldMinimumZ = {_usd_number(z0)}",
        f"    custom double kcg:worldMaximumZ = {_usd_number(z1)}",
    ]
    segment_index = 0
    for interval_index, (angle0, angle1) in enumerate(intervals_rad):
        for part_index, (part0, part1) in enumerate(_split_arc(angle0, angle1)):
            lines += _indent(
                _convex_arc_mesh_lines(
                    name=f"{segment_prefix}_{segment_index:03d}",
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    z0=z0,
                    z1=z1,
                    angle0_rad=part0,
                    angle1_rad=part1,
                    collision_role=collision_role,
                    trace_label=f"{name}:{interval_index}:{part_index}",
                    color=color,
                )
            )
            segment_index += 1
    lines += [f"    custom int kcg:exactConvexSegmentCount = {segment_index}", "}"]
    return lines


def _solid_intervals_outside_keyways(
    *,
    angles_deg: Sequence[float],
    widths_m: Sequence[float],
    reference_radius_m: float,
) -> tuple[list[tuple[float, float]], list[dict[str, float]]]:
    if len(angles_deg) != len(widths_m):
        raise ValueError("keyway angle/width count differs")
    openings: list[tuple[float, float]] = []
    evidence: list[dict[str, float]] = []
    full = 2.0 * math.pi
    for angle_deg, width_m in zip(angles_deg, widths_m):
        half = math.asin(0.5 * width_m / reference_radius_m)
        center = math.radians(angle_deg) % full
        start, stop = center - half, center + half
        evidence.append(
            {
                "center_deg": angle_deg,
                "width_m": width_m,
                "half_angle_deg": math.degrees(half),
            }
        )
        if start < 0.0:
            openings.extend(((0.0, stop), (start + full, full)))
        elif stop > full:
            openings.extend(((start, full), (0.0, stop - full)))
        else:
            openings.append((start, stop))
    merged: list[list[float]] = []
    for start, stop in sorted(openings):
        if merged and start <= merged[-1][1] + 1.0e-12:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    solids: list[tuple[float, float]] = []
    cursor = 0.0
    for start, stop in merged:
        if start > cursor + 1.0e-12:
            solids.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < full - 1.0e-12:
        solids.append((cursor, full))
    return solids, evidence


def _cube_lines(
    *,
    name: str,
    center: Sequence[float],
    dimensions: Sequence[float],
    rotate_z_deg: float,
    collision_role: str,
    trace_label: str,
    color: Sequence[float],
) -> list[str]:
    return [
        f'def Cube "{name}" (',
        '    apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
        ")",
        "{",
        "    double size = 1",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
        f"    custom string kcg:traceLabel = {_usd_string(trace_label)}",
        "    bool physics:collisionEnabled = 1",
        *_fine_connector_collision_attribute_lines(),
        f"    color3f[] primvars:displayColor = [{_usd_vec(color)}]",
        f"    double xformOp:rotateZ = {_usd_number(rotate_z_deg)}",
        f"    double3 xformOp:scale = {_usd_vec(dimensions)}",
        f"    double3 xformOp:translate = {_usd_vec(center)}",
        '    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]',
        "}",
    ]


def _cylinder_lines(
    *,
    name: str,
    center: Sequence[float],
    radius: float,
    height: float,
    collision_role: str,
    collision_enabled: bool,
    trace_label: str | None,
    color: Sequence[float],
) -> list[str]:
    schema = (
        ' (\n    apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]\n)'
        if collision_enabled
        else ""
    )
    lines = [f'def Cylinder "{name}"{schema}', "{"]
    lines += [
        '    uniform token axis = "Z"',
        f"    double radius = {_usd_number(radius)}",
        f"    double height = {_usd_number(height)}",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
    ]
    if trace_label is not None:
        lines.append(f"    custom string kcg:traceLabel = {_usd_string(trace_label)}")
    if collision_enabled:
        lines.append("    bool physics:collisionEnabled = 1")
        lines.extend(_fine_connector_collision_attribute_lines())
    else:
        lines.append("    custom bool kcg:dynamicCollisionEnabled = 0")
    lines += [
        f"    color3f[] primvars:displayColor = [{_usd_vec(color)}]",
        f"    double3 xformOp:translate = {_usd_vec(center)}",
        '    uniform token[] xformOpOrder = ["xformOp:translate"]',
        "}",
    ]
    return lines


def _sphere_lines(
    *,
    name: str,
    center: Sequence[float],
    radius: float,
    collision_role: str,
    trace_label: str,
    color: Sequence[float],
) -> list[str]:
    return [
        f'def Sphere "{name}" (',
        '    apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
        ")",
        "{",
        f"    double radius = {_usd_number(radius)}",
        f"    custom string kcg:collisionRole = {_usd_string(collision_role)}",
        f"    custom string kcg:traceLabel = {_usd_string(trace_label)}",
        "    custom string kcg:materialRole = \"coupling_bearing_and_shoulder\"",
        "    bool physics:collisionEnabled = 1",
        *_fine_connector_collision_attribute_lines(),
        f"    color3f[] primvars:displayColor = [{_usd_vec(color)}]",
        f"    double3 xformOp:translate = {_usd_vec(center)}",
        '    uniform token[] xformOpOrder = ["xformOp:translate"]',
        "}",
    ]


def _nut_body_shoulder_body0_lines(bearing: Mapping[str, Any]) -> list[str]:
    geometry = bearing["physical_shoulder_collision_geometry"]
    lines = [
        'def Xform "NutBearingShoulders"',
        "{",
        '    custom string kcg:representation = "physical_hard_stop"',
        '    custom string kcg:materialRole = "coupling_bearing_and_shoulder"',
    ]
    for sign, name in (("positive", "PositiveStop"), ("negative", "NegativeStop")):
        stop = geometry[f"{sign}_transZ_stop"]
        cap = _cylinder_lines(
            name="AnalyticCap",
            center=(0.0, 0.0, float(stop["body0_collider_center_local_z_m"])),
            radius=float(geometry["body0_cap_radius_m"]),
            height=float(geometry["body0_cap_axial_thickness_m"]),
            collision_role="hard_nut_body_shoulder",
            collision_enabled=True,
            trace_label=f"shoulder_{sign}_body0",
            color=(0.42, 0.46, 0.52),
        )
        cap[-1:-1] = [
            '    custom string kcg:materialRole = "coupling_bearing_and_shoulder"'
        ]
        lines += _indent(
            [
                f'def Xform "{name}"',
                "{",
                f"    custom double kcg:expectedContactTransZM = {_usd_number(stop['expected_contact_transZ_m'])}",
            ]
        )
        lines += _indent(cap, 2)
        lines += _indent(["}"], 1)
    lines += ["}"]
    return lines


def _nut_body_shoulder_body1_lines(bearing: Mapping[str, Any]) -> list[str]:
    geometry = bearing["physical_shoulder_collision_geometry"]
    phases_deg = [float(value) for value in geometry["body1_sphere_phases_deg"]]
    if len(phases_deg) != int(geometry["body1_sphere_count_per_direction"]):
        raise ValueError("physical shoulder sphere count differs from phase count")
    distribution_radius = float(geometry["body1_sphere_distribution_radius_m"])
    lines = [
        'def Xform "NutBearingShoulders"',
        "{",
        '    custom string kcg:representation = "physical_hard_stop"',
        '    custom string kcg:materialRole = "coupling_bearing_and_shoulder"',
    ]
    for sign, name in (("positive", "PositiveStop"), ("negative", "NegativeStop")):
        stop = geometry[f"{sign}_transZ_stop"]
        lines += _indent(
            [
                f'def Xform "{name}"',
                "{",
                f"    custom double kcg:expectedContactTransZM = {_usd_number(stop['expected_contact_transZ_m'])}",
            ]
        )
        for index, phase_deg in enumerate(phases_deg):
            phase_rad = math.radians(phase_deg)
            lines += _indent(
                _sphere_lines(
                    name=f"AnalyticSphere_{index}",
                    center=(
                        distribution_radius * math.cos(phase_rad),
                        distribution_radius * math.sin(phase_rad),
                        float(stop["body1_sphere_center_local_z_m"]),
                    ),
                    radius=float(geometry["body1_sphere_radius_m"]),
                    collision_role="hard_nut_body_shoulder",
                    trace_label=f"shoulder_{sign}_body1_{index}",
                    color=(0.48, 0.5, 0.56),
                ),
                2,
            )
        lines += _indent(["}"], 1)
    lines += ["}"]
    return lines


def _mass_attributes(body: Mapping[str, Any], *, kinematic: bool) -> list[str]:
    return [
        f"float physics:mass = {_usd_number(body['mass_kg'])}",
        f"point3f physics:centerOfMass = {_usd_vec(body['center_of_mass_m'])}",
        f"float3 physics:diagonalInertia = {_usd_vec(body['diagonal_inertia_kg_m2'])}",
        f"quatf physics:principalAxes = {_usd_vec(body['principal_axes_wxyz'])}",
        "bool physics:rigidBodyEnabled = 1",
        f"bool physics:kinematicEnabled = {_usd_bool(kinematic)}",
    ]


def _pair_paths(root: str, label: str) -> dict[str, str]:
    return {
        "pin": f"{root}/FixedReceptacle/Contacts/Pin_{label}",
        "barrier": f"{root}/FixedReceptacle/Contacts/Barrier_{label}",
        "socket": f"{root}/LoosePlug/BodyAssembly/Contacts/Socket_{label}",
        "smooth_entry": f"{root}/PairEffects/SmoothEntry_{label}",
        "deep_pin": f"{root}/PairEffects/DeepPin_{label}",
        "pin_isolation": f"{root}/PairEffects/PinIsolation_{label}",
    }


def _assembly_control_asset_text(
    document: Mapping[str, Any],
    *,
    contract_sha: str,
    physical_contract_sha: str,
    physical_shoulder_recipe: Mapping[str, Any],
    generator_sha: str,
    include_coupling_nut_grasp_collision: bool = False,
    authorized_overrides_v2: Mapping[str, Any] | None = None,
    authorized_overrides_v2_sha256: str | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    representation = "D38999_ASSEMBLY_CONTROL_V1"
    representation_root = f"{MULTILAYER_PARENT_PRIM}/{representation}"
    pair_root = f"{representation_root}/{PAIR_NAME}"
    contact_layout = document["contact_layout"]
    scale = float(contact_layout["coordinate_scale_m_per_in"])
    pairs: list[dict[str, Any]] = []
    for row in contact_layout["pairs"]:
        label = str(row["label"])
        center_in = [float(value) for value in row["center_in"]]
        center_m = [value * scale for value in center_in]
        paths = _pair_paths(pair_root, label)
        pairs.append(
            {
                "label": label,
                "center_in": center_in,
                "center_m": center_m,
                "pin_axis_fixed_local": [0.0, 0.0, 1.0],
                "socket_axis_plug_local": [0.0, 0.0, 1.0],
                "same_label_only": True,
                "paths": paths,
            }
        )

    if authorized_overrides_v2 is not None:
        if authorized_overrides_v2_sha256 is None:
            raise ValueError("V2 override fingerprint is required")
        include_coupling_nut_grasp_collision = bool(
            authorized_overrides_v2["initial_nut_grasp_collision_enabled"]
        )
        grasp_local_z_interval_m = tuple(
            float(value)
            for value in authorized_overrides_v2[
                "initial_nut_grasp_local_z_interval_m"
            ]
        )
        grasp_outer_radius_m = float(
            authorized_overrides_v2["initial_nut_grasp_outer_radius_m"]
        )
    else:
        grasp_local_z_interval_m = AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M
        grasp_outer_radius_m = AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M

    lines = _stage_header()
    lines += ['def Xform "World"', "{"]
    lines += _indent(['def Xform "D38999MultilayerV1"', "{"])
    lines += _indent([f'def Xform "{representation}"', "{"], 2)
    lines += _indent(
        _representation_metadata(
            representation=representation,
            contract_sha=contract_sha,
            generator_sha=generator_sha,
        ),
        3,
    )
    if authorized_overrides_v2 is not None:
        lines += _indent(
            [
                f"custom int kcg:eventOnsetProxyVersion = {int(authorized_overrides_v2['event_onset_proxy_version'])}",
                "custom string kcg:authorizedOverridesPath = "
                + _usd_string(AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix()),
                "custom string kcg:authorizedOverridesSha256 = "
                + _usd_string(authorized_overrides_v2_sha256),
                "custom bool kcg:initialNutGraspCollisionEnabled = "
                + _usd_bool(include_coupling_nut_grasp_collision),
                "custom bool kcg:newBackshellEnabled = "
                + _usd_bool(bool(authorized_overrides_v2["new_backshell_enabled"])),
            ],
            3,
        )
    truth = document["truth_firewall"]
    requirements = document["representation_requirements"][representation]
    lines += _indent(
        [
            "custom string kcg:representationPurpose = \"assembly_control\"",
            "custom bool kcg:sameLabelEffectOnly = 1",
            "custom bool kcg:crossLabelEffectAllowed = 0",
            "custom bool kcg:segmentedSocketEntryMicroWedgesAllowed = 0",
            "custom bool kcg:segmentedPinBarrierMicroWedgesAllowed = 0",
            "custom bool kcg:allCombinationMicroContactAllowed = 0",
            "custom bool kcg:magneticAttractionAllowed = 0",
            "custom bool kcg:postPhysicsPoseWriteAllowed = 0",
            "custom bool kcg:controllerTruthInputAllowed = 0",
            "custom bool kcg:modelInternalRelativeDisplacementAllowed = "
            + _usd_bool(truth["model_internal_relative_displacement_for_force_allowed"]),
            "custom bool kcg:nutBodyPhysicalShoulderEnabled = 1",
            "custom string kcg:physicalShoulderSourcePath = "
            + _usd_string(PHYSICAL_CONTRACT_RELATIVE_PATH.as_posix()),
            "custom string kcg:physicalShoulderSourceSha256 = "
            + _usd_string(physical_contract_sha),
            "custom int kcg:traceablePairCount = 61",
            "custom string[] kcg:continuousRealCollisionRoles = "
            + _usd_string_array(requirements["continuous_real_collision_required_for"]),
            "",
            f'def Xform "{PAIR_NAME}"',
            "{",
        ],
        3,
    )

    bodies = document["mass_properties"]["bodies"]
    keying = document["keying"]
    geometry = keying["nominal_collision_geometry"]
    events_by_name = {
        str(row["name"]): float(row["nominal_separation_m"])
        for row in document["assembly_events"]["ordered"]
    }
    first_event_m = events_by_name["five_key_polarization"]
    lines += _indent(
        [
            'def Xform "FixedReceptacle" (',
            '    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            ")",
            "{",
        ],
        4,
    )
    lines += _indent(_mass_attributes(bodies["fixed_receptacle"], kinematic=True), 5)
    lines += _indent(['def Xform "MatingShell"', "{"], 5)
    keyway_angles = [float(value) for value in keying["canonical_angles_deg"]]
    main_keyway_width = float(keying["main_keyway_width_mm"]["basic"]) * 1.0e-3
    minor_keyway_width = float(keying["minor_keyway_width_mm"]["basic"]) * 1.0e-3
    keyway_r0 = float(geometry["receptacle_clear_bore_radius_m"])
    keyway_r1 = float(geometry["keyway_slot_end_radius_m"])
    receptacle_outer_radius = float(geometry["receptacle_outer_radius_m"])
    keyway_local_z0, keyway_local_z1 = [
        float(value) for value in geometry["keyway_local_z_interval_m"]
    ]
    keyway_z0, keyway_z1 = -keyway_local_z1, -keyway_local_z0
    keyway_widths = [main_keyway_width] + [minor_keyway_width] * 4
    solid_intervals, keyway_opening_evidence = _solid_intervals_outside_keyways(
        angles_deg=keyway_angles,
        widths_m=keyway_widths,
        reference_radius_m=keyway_r0,
    )
    lines += _indent(['def Xform "ContinuousGuideRing"', "{"], 6)
    lines += _indent(
        [
            'custom string kcg:surfaceClass = "continuous_guide_with_five_physical_keyway_openings"',
            f"custom int kcg:keywayOpeningCount = {len(keyway_opening_evidence)}",
            "custom double[] kcg:keywayOpeningHalfAnglesDeg = ["
            + ", ".join(
                _usd_number(row["half_angle_deg"])
                for row in keyway_opening_evidence
            )
            + "]",
        ],
        7,
    )
    lines += _indent(
        _annular_segment_group_lines(
            name="OuterBackbone",
            inner_radius=keyway_r1,
            outer_radius=receptacle_outer_radius,
            z0=keyway_z0,
            z1=keyway_z1,
            collision_role="continuous_shell_and_guidance",
            color=(0.26, 0.43, 0.64),
            segment_prefix="OuterSegment",
        ),
        7,
    )
    lines += _indent(
        _annular_segment_group_lines(
            name="BoreGuideBetweenKeyways",
            inner_radius=keyway_r0,
            outer_radius=keyway_r1,
            z0=keyway_z0,
            z1=keyway_z1,
            collision_role="continuous_shell_and_guidance",
            color=(0.26, 0.43, 0.64),
            intervals_rad=solid_intervals,
            segment_prefix="BoreSegment",
        ),
        7,
    )
    lines += _indent(["}"], 6)
    lines += _indent(['def Xform "KeywayShell"', "{"], 6)
    for index, angle in enumerate(keyway_angles):
        width = main_keyway_width if index == 0 else minor_keyway_width
        theta = math.radians(angle)
        radial = (math.cos(theta), math.sin(theta))
        tangent = (-math.sin(theta), math.cos(theta))
        for side_name, sign in (("LeftWall", -1.0), ("RightWall", 1.0)):
            radius = 0.5 * (keyway_r0 + keyway_r1)
            center = (
                radius * radial[0] + sign * 0.5 * width * tangent[0],
                radius * radial[1] + sign * 0.5 * width * tangent[1],
                0.5 * (keyway_z0 + keyway_z1),
            )
            lines += _indent(
                _cube_lines(
                    name=f"Keyway_{index}_{side_name}",
                    center=center,
                    dimensions=(keyway_r1 - keyway_r0, 0.00012, keyway_z1 - keyway_z0),
                    rotate_z_deg=angle,
                    collision_role="continuous_keyway_wall",
                    trace_label=f"keyway_{index}",
                    color=(0.25, 0.42, 0.62),
                ),
                7,
            )
    lines += _indent(["}"], 6)
    stop = document["metal_stop"]
    fixed_stop_surface = float(stop["receptacle_stop_surface_depth_from_receptacle_B_m"])
    fixed_stop_thickness = float(
        stop["assembly_control_collision"]["fixed_cap_axial_thickness_m"]
    )
    lines += _indent(
        _annular_segment_group_lines(
            name="MetalStop",
            inner_radius=float(stop["assembly_control_collision"]["plug_stop_distribution_radius_m"]) - 0.001,
            outer_radius=float(stop["assembly_control_collision"]["fixed_cap_radius_m"]),
            z0=fixed_stop_surface - fixed_stop_thickness,
            z1=fixed_stop_surface,
            collision_role="continuous_real_metal_stop_fixed",
            color=(0.58, 0.6, 0.64),
            segment_prefix="StopSegment",
        ),
        6,
    )
    lines += _indent(["}"], 5)
    lines += _indent(['def Xform "Contacts"', "{"], 5)
    pin_radius = 0.5 * float(
        document["major_dimensions"]["size20_contact_interface"]["pin_engaging_diameter_in"]["basic"]
    ) * 0.0254
    pin_tip_z = float(
        document["axial_interface"]["receptacle_depths_from_datum_B_mm"]["pin_tip_M_nominal_proxy"]
    ) * 1.0e-3
    pin_length = 0.004
    for pair in pairs:
        x_value, y_value = pair["center_m"]
        label = pair["label"]
        lines += _indent(
            _cylinder_lines(
                name=f"Pin_{label}",
                center=(x_value, y_value, pin_tip_z + 0.5 * pin_length),
                radius=pin_radius,
                height=pin_length,
                collision_role="same_label_analytic_pin_geometry",
                collision_enabled=False,
                trace_label=label,
                color=(0.78, 0.58, 0.18),
            ),
            6,
        )
        lines += _indent(
            [
                f'def Xform "Barrier_{label}"',
                "{",
                f"    custom string kcg:traceLabel = {_usd_string(label)}",
                "    custom string kcg:representation = \"continuous_internal_axial_response_only\"",
                "    custom bool kcg:segmentedCollisionEnabled = 0",
                "}",
            ],
            6,
        )
    lines += _indent(["}"], 5)
    lines += _indent(
        [
            'def Xform "PeripheralSeal"',
            "{",
            "    custom string kcg:modelType = \"continuous_annular_axial_spring_damper\"",
            f"    custom double kcg:firstTouchSeparationM = {_usd_number(document['elastic_contact_models']['peripheral_seal']['first_touch_separation_m'])}",
            f"    custom double kcg:stiffnessNM = {_usd_number(document['elastic_contact_models']['peripheral_seal']['aggregate_stiffness_n_m'])}",
            f"    custom double kcg:dampingNSM = {_usd_number(document['elastic_contact_models']['peripheral_seal']['aggregate_damping_n_s_m'])}",
            "    custom bool kcg:controllerVisible = 0",
            "}",
            "}",
        ],
        5,
    )

    lines += _indent(['def Xform "LoosePlug"', "{"], 4)
    lines += _indent(
        [
            'def Xform "BodyAssembly" (',
            '    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            ")",
            "{",
        ],
        5,
    )
    lines += _indent(_mass_attributes(bodies["loose_plug_body_assembly"], kinematic=False), 6)
    shell_r0, shell_r1 = [float(value) for value in geometry["plug_shell_radial_interval_m"]]
    key_z0, key_z1 = [float(value) for value in geometry["plug_key_local_depth_interval_m"]]
    lines += _indent(['def Xform "MatingShell"', "{"], 6)
    lines += _indent(
        _ring_mesh_lines(
            name="ContinuousPlugGuide",
            inner_radius=shell_r0,
            outer_radius=shell_r1,
            z0=first_event_m,
            z1=float(stop["nominal_bottoming_separation_m"]),
            collision_role="continuous_shell_and_guidance",
            color=(0.54, 0.56, 0.6),
            collision_enabled=False,
        ),
        7,
    )
    lines += _indent(
        _annular_segment_group_lines(
            name="ContinuousPlugGuideCollision",
            inner_radius=shell_r0,
            outer_radius=shell_r1,
            z0=first_event_m,
            z1=float(stop["nominal_bottoming_separation_m"]),
            collision_role="continuous_shell_and_guidance",
            color=(0.54, 0.56, 0.6),
            segment_prefix="GuideSegment",
        ),
        7,
    )
    lines += _indent(["}"], 6)
    lines += _indent(['def Xform "PolarizingKeys"', "{"], 6)
    authored_key_angles = [
        float(value) for value in keying["plug_local_authored_angles_deg"]["values"]
    ]
    collision_key_angles = [(-value) % 360.0 for value in authored_key_angles]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(collision_key_angles, keyway_angles)
    ):
        raise ValueError("plug-local key angles do not map to canonical mating angles")
    key_r0, key_r1 = [float(value) for value in geometry["plug_key_radial_interval_m"]]
    main_key_width = float(keying["main_key_width_mm"]["basic"]) * 1.0e-3
    minor_key_width = float(keying["minor_key_width_mm"]["basic"]) * 1.0e-3
    for index, angle in enumerate(collision_key_angles):
        theta = math.radians(angle)
        radius = 0.5 * (key_r0 + key_r1)
        width = main_key_width if index == 0 else minor_key_width
        lines += _indent(
            _cube_lines(
                name=f"Key_{index}",
                center=(radius * math.cos(theta), radius * math.sin(theta), 0.5 * (key_z0 + key_z1)),
                dimensions=(key_r1 - key_r0, width, key_z1 - key_z0),
                rotate_z_deg=angle,
                collision_role="continuous_polarizing_key",
                trace_label=f"key_{index}",
                color=(0.54, 0.56, 0.6),
            ),
            7,
        )
    lines += _indent(["}"], 6)
    lines += _indent(['def Xform "InternalMatingShell"', "{"], 6)
    plug_stop_thickness = 0.0002
    plug_stop_surface = float(stop["plug_stop_surface_depth_from_plug_B_m"])
    lines += _indent(
        _cylinder_lines(
            name="MetalStop",
            center=(0.0, 0.0, plug_stop_surface + 0.5 * plug_stop_thickness),
            radius=float(stop["assembly_control_collision"]["plug_stop_distribution_radius_m"]),
            height=plug_stop_thickness,
            collision_role="continuous_real_metal_stop_plug",
            collision_enabled=True,
            trace_label=None,
            color=(0.58, 0.6, 0.64),
        ),
        7,
    )
    lines += _indent(["}"], 6)
    lines += _indent(['def Xform "Contacts"', "{"], 6)
    socket_face_z = float(
        document["axial_interface"]["plug_depths_from_datum_B_mm"]["socket_insert_face_B_nominal_proxy"]
    ) * 1.0e-3
    socket_radius = 0.5 * float(
        document["major_dimensions"]["size20_contact_interface"]["socket_entry_F_diameter_mm"]["nominal"]
    ) * 1.0e-3
    for pair in pairs:
        x_value, y_value = pair["center_m"]
        label = pair["label"]
        lines += _indent(
            [
                f'def Xform "Socket_{label}"',
                "{",
                f"    custom string kcg:traceLabel = {_usd_string(label)}",
                f"    custom double2 kcg:centerM = {_usd_vec((x_value, y_value))}",
                "    custom string kcg:entryGuideType = \"continuous_smooth_axisymmetric_analytic_funnel\"",
                f"    custom double kcg:entryRadiusM = {_usd_number(socket_radius)}",
                f"    custom double kcg:entryPlaneLocalZM = {_usd_number(socket_face_z)}",
                "    custom bool kcg:segmentedMicroWedgeCollisionEnabled = 0",
                "    custom bool kcg:controllerVisible = 0",
                "}",
            ],
            7,
        )
    lines += _indent(["}"], 6)
    lines += _indent(_nut_body_shoulder_body0_lines(physical_shoulder_recipe), 6)
    lines += _indent(["}"], 5)
    lines += _indent(
        [
            'def Xform "CouplingNut" (',
            '    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            ")",
            "{",
        ],
        5,
    )
    lines += _indent(_mass_attributes(bodies["coupling_nut"], kinematic=False), 6)
    lines += _indent(
        [
            "custom string kcg:threadConstraintType = \"explicit_rotation_translation\"",
            f"custom int kcg:threadStarts = {int(document['thread']['starts'])}",
            f"custom double kcg:threadLeadMPerRevolution = {_usd_number(float(document['thread']['lead_mm_per_revolution']) * 1.0e-3)}",
            "custom bool kcg:jointDriveAllowed = 0",
            "custom bool kcg:magneticMechanismAllowed = 0",
        ],
        6,
    )
    lines += _indent(_nut_body_shoulder_body1_lines(physical_shoulder_recipe), 6)
    if include_coupling_nut_grasp_collision:
        grasp_lines = _cylinder_lines(
            name="CouplingNutGraspCollision",
            center=(
                0.0,
                0.0,
                0.5
                * (
                    grasp_local_z_interval_m[0]
                    + grasp_local_z_interval_m[1]
                ),
            ),
            radius=grasp_outer_radius_m,
            height=(
                grasp_local_z_interval_m[1]
                - grasp_local_z_interval_m[0]
            ),
            collision_role="coupling_nut_grasp_collision",
            collision_enabled=True,
            trace_label="INITIAL_PICK_AND_TRANSPORT_GRASP_REGION",
            color=(0.45, 0.47, 0.52),
        )
        grasp_lines[-1:-1] = [
            '    uniform token purpose = "guide"',
            "    custom bool kcg:externalFingerContactOnly = 1",
            "    custom bool kcg:solidCoreMayContactBodyAssembly = 0",
            "    custom string kcg:collisionFilteringOwner = \"tabletop_grasp_runner_pre_reset\"",
            f"    custom double2 kcg:authorizedLocalZIntervalM = {_usd_vec(grasp_local_z_interval_m)}",
            f"    custom double kcg:authorizedOuterRadiusM = {_usd_number(grasp_outer_radius_m)}",
        ]
        lines += _indent(grasp_lines, 6)
    lines += _indent(["}"], 6)
    joint = document["coupling_nut_motion"]
    lines += _indent(
        [
            'def PhysicsJoint "CouplingNutJoint" (',
            '    prepend apiSchemas = ["PhysicsLimitAPI:transX", "PhysicsLimitAPI:transY", "PhysicsLimitAPI:transZ", "PhysicsLimitAPI:rotX", "PhysicsLimitAPI:rotY"]',
            ")",
            "{",
            f"    rel physics:body0 = <{pair_root}/LoosePlug/BodyAssembly>",
            f"    rel physics:body1 = <{pair_root}/LoosePlug/CouplingNut>",
            "    bool physics:collisionEnabled = 1",
            "    bool physics:jointEnabled = 1",
            "    float limit:transX:physics:high = -1",
            "    float limit:transX:physics:low = 1",
            "    float limit:transY:physics:high = -1",
            "    float limit:transY:physics:low = 1",
            "    float limit:rotX:physics:high = -1",
            "    float limit:rotX:physics:low = 1",
            "    float limit:rotY:physics:high = -1",
            "    float limit:rotY:physics:low = 1",
            f"    float limit:transZ:physics:low = {_usd_number(joint['transZ_backup_limits_m']['low'])}",
            f"    float limit:transZ:physics:high = {_usd_number(joint['transZ_backup_limits_m']['high'])}",
            "    custom string kcg:freeDegreeOfFreedom = \"rotZ\"",
            "    custom bool kcg:softwareAxialPoseWriteAllowed = 0",
            "}",
            "}",
        ],
        5,
    )

    effects = document["elastic_contact_models"]
    lines += _indent(['def Scope "PairEffects"', "{"], 4)
    for pair in pairs:
        label = pair["label"]
        paths = pair["paths"]
        common = [
            f"custom string kcg:traceLabel = {_usd_string(label)}",
            f"rel kcg:pin = <{paths['pin']}>",
            f"rel kcg:socket = <{paths['socket']}>",
            "custom bool kcg:sameLabelOnly = 1",
            "custom bool kcg:controllerVisible = 0",
        ]
        lines += _indent(
            [
                f'def Scope "SmoothEntry_{label}"',
                "{",
                *["    " + row for row in common],
                "    custom string kcg:modelType = \"continuous_smooth_axisymmetric_radial_guide\"",
                f"    custom double kcg:maximumPhysicalDeflectionM = {_usd_number(effects['socket_contact_per_label']['maximum_physical_deflection_m'])}",
                "}",
                f'def Scope "DeepPin_{label}"',
                "{",
                *["    " + row for row in common],
                "    custom string kcg:modelType = \"same_label_radial_spring_damper\"",
                f"    custom double kcg:stiffnessNM = {_usd_number(effects['socket_contact_per_label']['aggregate_stiffness_n_m'])}",
                f"    custom double kcg:dampingNSM = {_usd_number(effects['socket_contact_per_label']['aggregate_damping_n_s_m'])}",
                f"    custom double kcg:nominalInterferenceM = {_usd_number(effects['socket_contact_per_label']['nominal_radial_interference_m'])}",
                "}",
                f'def Scope "PinIsolation_{label}"',
                "{",
                *["    " + row for row in common],
                f"    rel kcg:barrier = <{paths['barrier']}>",
                "    custom string kcg:modelType = \"same_label_continuous_axial_spring_damper\"",
                f"    custom double kcg:firstTouchSeparationM = {_usd_number(effects['pin_isolation_seal_per_label']['nominal_first_touch_separation_m'])}",
                f"    custom double kcg:stiffnessNM = {_usd_number(effects['pin_isolation_seal_per_label']['per_label_effective_stiffness_n_m'])}",
                f"    custom double kcg:dampingNSM = {_usd_number(effects['pin_isolation_seal_per_label']['per_label_effective_damping_n_s_m'])}",
                "    custom bool kcg:axialTravelUsedAsNormalDeflection = 0",
                "}",
            ],
            5,
        )
    lines += _indent(["}"], 4)

    thread = document["thread"]
    detent = document["anti_decoupling"]
    lines += _indent(
        [
            'def Scope "MotionRelations"',
            "{",
            '    def Scope "ThreeStartThread"',
            "    {",
            f"        custom int kcg:starts = {int(thread['starts'])}",
            f"        custom double kcg:leadMPerRevolution = {_usd_number(float(thread['lead_mm_per_revolution']) * 1.0e-3)}",
            f"        custom double[] kcg:startPhasesDeg = [{', '.join(_usd_number(v) for v in thread['start_phases_deg'])}]",
            f"        custom string kcg:positiveInsertionRelation = {_usd_string(thread['control_representation']['positive_insertion_relation'])}",
            "        custom bool kcg:runtimeEngagementSwitchAllowed = 0",
            "        custom bool kcg:softwareAxialPoseWriteAllowed = 0",
            "    }",
            '    def Scope "AntiDecoupling"',
            "    {",
            "        custom string kcg:modelType = \"bounded_directional_periodic_resistance\"",
            f"        custom int kcg:cyclesPerRevolution = {int(detent['cycle_count_per_revolution'])}",
            f"        custom int kcg:followerCount = {int(detent['follower_count'])}",
            f"        custom double kcg:meanRadiusM = {_usd_number(detent['mean_radius_m'])}",
            f"        custom double kcg:perFollowerStiffnessNM = {_usd_number(detent['per_follower_stiffness_n_m'])}",
            f"        custom double kcg:perFollowerDampingNSM = {_usd_number(detent['per_follower_damping_n_s_m'])}",
            "        custom bool kcg:magneticMechanismAllowed = 0",
            "    }",
            "}",
        ],
        4,
    )
    lines += _indent(['def Scope "AssemblyEvents"', "{"], 4)
    for event in document["assembly_events"]["ordered"]:
        lines += _indent(
            [
                f'def Scope "Event_{int(event["ordinal"]):02d}_{event["name"]}"',
                "{",
                f"    custom int kcg:ordinal = {int(event['ordinal'])}",
                f"    custom string kcg:eventName = {_usd_string(event['name'])}",
                f"    custom double kcg:nominalSeparationM = {_usd_number(event['nominal_separation_m'])}",
                "    custom bool kcg:controllerSwitchInputAllowed = 0",
                "}",
            ],
            5,
        )
    lines += _indent(["}"], 4)
    lines += _indent(["}"], 3)
    lines += _indent(["}"], 2)
    lines += _indent(["}"], 1)
    lines += ["}", ""]

    logical = document["logical_path_interface"]["logical_parts"]
    logical_paths = {
        name: f"{pair_root}/{suffix}" for name, suffix in logical.items() if name != "pair_root"
    }
    logical_paths["pair_root"] = pair_root
    return "\n".join(lines), pairs, logical_paths


def _validate_usda_text(text: str, representation: str) -> dict[str, Any]:
    checks = {
        "header": text.startswith("#usda 1.0\n"),
        "balanced_braces": text.count("{") == text.count("}"),
        "representation_id_present": representation in text,
        "contract_sha_present": EXPECTED_CONTRACT_SHA256 in text,
        "formal_r12_false": "custom bool kcg:formalR12Frozen = 0" in text,
        "hardware_authorized_false": "custom bool kcg:hardwareAuthorized = 0" in text,
    }
    if not all(checks.values()):
        raise ValueError(f"generated USDA validation failed for {representation}: {checks}")
    return checks


def _build_documents(
    document: Mapping[str, Any],
    *,
    output_root: Path,
    contract_sha: str,
    physical_contract_sha: str,
    physical_shoulder_recipe: Mapping[str, Any],
    generator_sha: str,
    include_coupling_nut_grasp_collision: bool = False,
    authorized_overrides_v2: Mapping[str, Any] | None = None,
    authorized_overrides_v2_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    visual_source = document["representation_requirements"][
        "D38999_VISUAL_COMPLETE_V1"
    ]["visible_geometry_source"]
    local_source = document["representation_requirements"][
        "D38999_LOCAL_CONTACT_REFERENCE_V1"
    ]
    if visual_source["path"] != local_source["direct_reference_path"]:
        raise ValueError("visual and local representations must reference one frozen source")
    if visual_source["sha256"] != local_source["direct_reference_sha256"]:
        raise ValueError("visual and local source fingerprints differ")
    source_path = (WORKSPACE_ROOT / str(visual_source["path"])).resolve()
    if _sha256(source_path) != str(visual_source["sha256"]):
        raise PermissionError("frozen high-detail reference fingerprint changed")
    source_relative = Path(os.path.relpath(source_path, output_root)).as_posix()

    visual = _reference_asset_text(
        representation="D38999_VISUAL_COMPLETE_V1",
        source_relative_to_output=source_relative,
        source_repo_path=str(visual_source["path"]),
        source_sha=str(visual_source["sha256"]),
        contract_sha=contract_sha,
        generator_sha=generator_sha,
        read_only=True,
        complete_robot_mainline_allowed=False,
    )
    assembly, pairs, assembly_logical_paths = _assembly_control_asset_text(
        document,
        contract_sha=contract_sha,
        physical_contract_sha=physical_contract_sha,
        physical_shoulder_recipe=physical_shoulder_recipe,
        generator_sha=generator_sha,
        include_coupling_nut_grasp_collision=(
            include_coupling_nut_grasp_collision
        ),
        authorized_overrides_v2=authorized_overrides_v2,
        authorized_overrides_v2_sha256=authorized_overrides_v2_sha256,
    )
    local = _reference_asset_text(
        representation="D38999_LOCAL_CONTACT_REFERENCE_V1",
        source_relative_to_output=source_relative,
        source_repo_path=str(local_source["direct_reference_path"]),
        source_sha=str(local_source["direct_reference_sha256"]),
        contract_sha=contract_sha,
        generator_sha=generator_sha,
        read_only=True,
        complete_robot_mainline_allowed=False,
    )
    documents = {
        "D38999_VISUAL_COMPLETE_V1.usda": visual,
        "D38999_ASSEMBLY_CONTROL_V1.usda": assembly,
        "D38999_LOCAL_CONTACT_REFERENCE_V1.usda": local,
    }
    validation = {
        filename: _validate_usda_text(text, Path(filename).stem)
        for filename, text in documents.items()
    }
    assembly_forbidden_tokens = (
        "hard_socket_entries_61",
        "pin_barriers_61",
        "six_fixed_annular_wedge_petals_per_socket",
        "segmented_axisymmetric_pin_barrier_contact",
    )
    forbidden_hits = [token for token in assembly_forbidden_tokens if token in assembly]
    if forbidden_hits:
        raise ValueError(f"forbidden high-detail contact families leaked: {forbidden_hits}")
    labels = [pair["label"] for pair in pairs]
    if len(labels) != 61 or len(set(labels)) != 61:
        raise ValueError("assembly control pair mapping is not one-to-one")
    mapping = {
        "schema_version": "kcg_d38999_multilayer_mapping_v1",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "assembly_axis": document["coordinate_frames"]["assembly_axis"],
        "keying": {
            "count": int(document["keying"]["key_count"]),
            "canonical_angles_deg": document["keying"]["canonical_angles_deg"],
            "plug_local_authored_angles_deg": document["keying"][
                "plug_local_authored_angles_deg"
            ]["values"],
        },
        "mass_properties": document["mass_properties"],
        "assembly_events": document["assembly_events"]["ordered"],
        "thread": {
            "starts": int(document["thread"]["starts"]),
            "lead_mm_per_revolution": float(document["thread"]["lead_mm_per_revolution"]),
            "start_phases_deg": document["thread"]["start_phases_deg"],
        },
        "metal_stop": document["metal_stop"],
        "representations": {
            "D38999_VISUAL_COMPLETE_V1": {
                "root": document["logical_path_interface"]["representation_roots"][
                    "D38999_VISUAL_COMPLETE_V1"
                ],
                "pair_root": f"{MULTILAYER_PARENT_PRIM}/D38999_VISUAL_COMPLETE_V1/{PAIR_NAME}",
                "direct_reference": str(visual_source["path"]),
                "direct_reference_sha256": str(visual_source["sha256"]),
                "visible_geometry_preserved": True,
            },
            "D38999_ASSEMBLY_CONTROL_V1": {
                "root": document["logical_path_interface"]["representation_roots"][
                    "D38999_ASSEMBLY_CONTROL_V1"
                ],
                "pair_root": f"{MULTILAYER_PARENT_PRIM}/D38999_ASSEMBLY_CONTROL_V1/{PAIR_NAME}",
                "logical_paths": assembly_logical_paths,
                "continuous_real_collision_roles": document["representation_requirements"][
                    "D38999_ASSEMBLY_CONTROL_V1"
                ]["continuous_real_collision_required_for"],
                "pair_effects": pairs,
                "pair_effect_count": len(pairs),
                "cross_label_effects": [],
                "forbidden_high_detail_family_hits": forbidden_hits,
            },
            "D38999_LOCAL_CONTACT_REFERENCE_V1": {
                "root": document["logical_path_interface"]["representation_roots"][
                    "D38999_LOCAL_CONTACT_REFERENCE_V1"
                ],
                "pair_root": f"{MULTILAYER_PARENT_PRIM}/D38999_LOCAL_CONTACT_REFERENCE_V1/{PAIR_NAME}",
                "direct_reference": str(local_source["direct_reference_path"]),
                "direct_reference_sha256": str(local_source["direct_reference_sha256"]),
                "read_only": True,
                "allowed_tests": local_source["allowed_tests"],
                "complete_robot_assembly_mainline_allowed": False,
            },
        },
        "truth_firewall": document["truth_firewall"],
        "static_generation_checks": {
            "usda": validation,
            "one_contract": True,
            "one_generator": True,
            "traceable_pair_count": len(pairs),
            "cross_label_effect_count": 0,
            "segmented_socket_entry_micro_wedge_count": 0,
            "segmented_pin_barrier_micro_wedge_count": 0,
            "all_combination_micro_contact_count": 0,
            "simulation_started": False,
            "formal_p1_pass_claimed": False,
        },
    }
    return documents, mapping


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary file exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _guarded_event_onset_update(
    *,
    arguments: argparse.Namespace,
    document: Mapping[str, Any],
    output_root: Path,
    result_path: Path,
    documents: Mapping[str, str],
    contract_sha: str,
    generator_sha: str,
    verified_inputs: Mapping[str, str],
) -> int:
    state_path = WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    required_state = {
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION",
        "status": "IMPLEMENTING",
        "phase": "DYNAMIC_EVENT_ONSET_CALIBRATION",
        "assembly_control_representation_generation_authorized": True,
        "formal_contract_changes_authorized": False,
        "high_detail_assets_read_only": True,
    }
    mismatches = {
        key: {"actual": state.get(key), "expected": expected}
        for key, expected in required_state.items()
        if state.get(key) != expected
    }
    decision = state.get("human_dynamic_continuation_decision", {})
    if decision.get("event_onset_proxy_fix_authorized") is not True:
        mismatches["event_onset_proxy_fix_authorized"] = {
            "actual": decision.get("event_onset_proxy_fix_authorized"),
            "expected": True,
        }
    if mismatches:
        raise PermissionError(f"DYN-A1 event-onset update guard failed: {mismatches}")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    if result_path.exists():
        raise FileExistsError(f"event-onset build result already exists: {result_path}")

    assembly_name = "D38999_ASSEMBLY_CONTROL_V1.usda"
    assembly_path = output_root / assembly_name
    if _sha256(assembly_path) != EXPECTED_PRE_ONSET_ASSEMBLY_SHA256:
        raise PermissionError("pre-onset assembly-control fingerprint changed")
    preserved_before = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_before != EXPECTED_PRESERVED_OUTPUT_SHA256:
        raise PermissionError(
            "visual/local/mapping fingerprints differ before guarded update: "
            f"{preserved_before}"
        )
    backup_path = (WORKSPACE_ROOT / EVENT_ONSET_BACKUP_RELATIVE_PATH).resolve()
    if backup_path.exists():
        raise FileExistsError(f"event-onset backup already exists: {backup_path}")

    new_text = documents[assembly_name]
    new_checks = _validate_usda_text(new_text, "D38999_ASSEMBLY_CONTROL_V1")
    plan = {
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION",
        "hypothesis_id": "H-ONSET-01",
        "dry_run": bool(arguments.dry_run),
        "single_asset_write": _repo_relative(assembly_path),
        "backup_write": EVENT_ONSET_BACKUP_RELATIVE_PATH.as_posix(),
        "result_write": EVENT_ONSET_RESULT_RELATIVE_PATH.as_posix(),
        "preserved_byte_identical": sorted(EXPECTED_PRESERVED_OUTPUT_SHA256),
        "contract_sha256": contract_sha,
        "generator_sha256": generator_sha,
        "validation": new_checks,
        "simulation_will_start": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assembly_path, backup_path)
    if _sha256(backup_path) != EXPECTED_PRE_ONSET_ASSEMBLY_SHA256:
        raise RuntimeError("assembly-control backup fingerprint differs")
    _write_text_atomic(assembly_path, new_text)
    new_sha = _sha256(assembly_path)
    if new_sha == EXPECTED_PRE_ONSET_ASSEMBLY_SHA256:
        raise RuntimeError("guarded update did not change the assembly-control asset")
    preserved_after = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_after != preserved_before:
        raise RuntimeError(
            "guarded update changed visual/local/mapping output: "
            f"before={preserved_before}, after={preserved_after}"
        )
    protected_after = {
        CONTRACT_RELATIVE_PATH.as_posix(): _sha256(WORKSPACE_ROOT / CONTRACT_RELATIVE_PATH),
        **{
            relative: _sha256(WORKSPACE_ROOT / relative)
            for relative in verified_inputs
        },
    }
    expected_protected = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        **dict(verified_inputs),
    }
    if protected_after != expected_protected:
        raise RuntimeError("authoritative inputs changed during guarded update")

    geometry = document["keying"]["nominal_collision_geometry"]
    stop = document["metal_stop"]
    result = {
        "schema_version": "kcg_dyn_a1_event_onset_build_result_v1",
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION",
        "hypothesis_id": "H-ONSET-01",
        "generated_at_utc": _utc_now(),
        "outcome": "PASS",
        "classification": "ASSEMBLY_CONTROL_EVENT_ONSET_PROXY_REGENERATED",
        "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": contract_sha},
        "generator": {"path": GENERATOR_RELATIVE_PATH.as_posix(), "sha256": generator_sha},
        "assembly_control": {
            "path": _repo_relative(assembly_path),
            "sha256_before": EXPECTED_PRE_ONSET_ASSEMBLY_SHA256,
            "sha256_after": new_sha,
            "backup_path": EVENT_ONSET_BACKUP_RELATIVE_PATH.as_posix(),
            "backup_sha256": _sha256(backup_path),
        },
        "preserved_outputs_before": preserved_before,
        "preserved_outputs_after": preserved_after,
        "protected_authoritative_inputs_after": protected_after,
        "proxy_derivation": {
            "cube_full_size_semantics": True,
            "cube_xform_order": ["translate", "rotateZ", "scale"],
            "fixed_receptacle_depth_mapping": "shared_z=[-local_z1,-local_z0]",
            "plug_guide_front_m": float(
                document["assembly_events"]["ordered"][0]["nominal_separation_m"]
            ),
            "plug_guide_radial_clearance_m": float(
                geometry["receptacle_clear_bore_radius_m"]
            ) - float(geometry["plug_shell_radial_interval_m"][1]),
            "fixed_stop_volume_z_m": [
                float(stop["receptacle_stop_surface_depth_from_receptacle_B_m"])
                - float(stop["assembly_control_collision"]["fixed_cap_axial_thickness_m"]),
                float(stop["receptacle_stop_surface_depth_from_receptacle_B_m"]),
            ],
            "plug_stop_surface_m": float(stop["plug_stop_surface_depth_from_plug_B_m"]),
            "plug_stop_center_m": float(stop["plug_stop_surface_depth_from_plug_B_m"])
            + 0.0001,
            "convex_points_local_unit": "millimetre",
            "convex_xform_scale": LOCAL_MILLIMETRE_TO_METRE,
            "convex_min_thickness_local_mm": CONVEX_MIN_THICKNESS_LOCAL_MM,
        },
        "targeted_fix_count": 1,
        "onset_probe_run_count": 0,
        "visual_model_modified": False,
        "local_contact_reference_modified": False,
        "master_contract_modified": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
        "simulation_started": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _guarded_nut_grasp_update(
    *,
    arguments: argparse.Namespace,
    output_root: Path,
    result_path: Path,
    documents: Mapping[str, str],
    contract_sha: str,
    generator_sha: str,
    verified_inputs: Mapping[str, str],
) -> int:
    state_path = WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    required_state = {
        "task_id": "DYN-B2-B4-THREE-FINGER-GRASP",
        "status": "IMPLEMENTING",
        "phase": "DYNAMIC_THREE_FINGER_GRASP",
        "assembly_control_representation_generation_authorized": True,
        "formal_contract_changes_authorized": False,
        "high_detail_assets_read_only": True,
    }
    mismatches = {
        key: {"actual": state.get(key), "expected": expected}
        for key, expected in required_state.items()
        if state.get(key) != expected
    }
    decision = state.get("human_dynamic_continuation_decision", {})
    expected_decision = {
        "coupling_nut_rear_grasp_authorized": True,
        "coupling_nut_grasp_local_z_interval_m": list(
            AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M
        ),
        "coupling_nut_outer_radius_m": AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M,
        "new_backshell_authorized": False,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            mismatches[f"human_dynamic_continuation_decision.{key}"] = {
                "actual": decision.get(key),
                "expected": expected,
            }
    grasp_state = state.get("dynamic_red_gate", {}).get(
        "three_finger_grasp", {}
    )
    expected_grasp_state = {
        "status": "IMPLEMENTING",
        "authorized_local_z_interval_m": list(
            AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M
        ),
        "authorized_outer_radius_m": AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M,
        "new_backshell_created": False,
    }
    for key, expected in expected_grasp_state.items():
        if grasp_state.get(key) != expected:
            mismatches[f"dynamic_red_gate.three_finger_grasp.{key}"] = {
                "actual": grasp_state.get(key),
                "expected": expected,
            }
    if mismatches:
        raise PermissionError(f"DYN-B2-B4 nut-grasp update guard failed: {mismatches}")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    if result_path.exists():
        raise FileExistsError(f"nut-grasp build result already exists: {result_path}")

    assembly_name = "D38999_ASSEMBLY_CONTROL_V1.usda"
    assembly_path = output_root / assembly_name
    actual_before = _sha256(assembly_path)
    if actual_before != EXPECTED_PRE_GRASP_ASSEMBLY_SHA256:
        raise PermissionError(
            "pre-grasp assembly-control fingerprint changed: "
            f"actual={actual_before}, expected={EXPECTED_PRE_GRASP_ASSEMBLY_SHA256}"
        )
    preserved_before = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_before != EXPECTED_PRESERVED_OUTPUT_SHA256:
        raise PermissionError(
            "visual/local/mapping fingerprints differ before guarded update: "
            f"{preserved_before}"
        )
    backup_path = (WORKSPACE_ROOT / NUT_GRASP_BACKUP_RELATIVE_PATH).resolve()
    if backup_path.exists():
        raise FileExistsError(f"nut-grasp backup already exists: {backup_path}")

    new_text = documents[assembly_name]
    new_checks = _validate_usda_text(new_text, "D38999_ASSEMBLY_CONTROL_V1")
    required_tokens = {
        'custom string kcg:collisionRole = "coupling_nut_grasp_collision"': 1,
        "custom double2 kcg:authorizedLocalZIntervalM = (0.009, 0.029)": 1,
        "custom double kcg:authorizedOuterRadiusM = 0.024": 1,
        "custom bool kcg:solidCoreMayContactBodyAssembly = 0": 1,
    }
    token_counts = {token: new_text.count(token) for token in required_tokens}
    if token_counts != required_tokens:
        raise ValueError(
            "authorized nut-grasp collider tokens differ: "
            f"actual={token_counts}, expected={required_tokens}"
        )
    plan = {
        "task_id": "DYN-B2-B4-THREE-FINGER-GRASP",
        "dry_run": bool(arguments.dry_run),
        "single_asset_write": _repo_relative(assembly_path),
        "backup_write": NUT_GRASP_BACKUP_RELATIVE_PATH.as_posix(),
        "result_write": NUT_GRASP_RESULT_RELATIVE_PATH.as_posix(),
        "authorized_local_z_interval_m": list(
            AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M
        ),
        "authorized_outer_radius_m": AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M,
        "preserved_byte_identical": sorted(EXPECTED_PRESERVED_OUTPUT_SHA256),
        "validation": new_checks,
        "token_counts": token_counts,
        "simulation_will_start": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assembly_path, backup_path)
    if _sha256(backup_path) != EXPECTED_PRE_GRASP_ASSEMBLY_SHA256:
        raise RuntimeError("assembly-control grasp backup fingerprint differs")
    _write_text_atomic(assembly_path, new_text)
    new_sha = _sha256(assembly_path)
    if new_sha == EXPECTED_PRE_GRASP_ASSEMBLY_SHA256:
        raise RuntimeError("guarded nut-grasp update did not change the asset")
    preserved_after = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_after != preserved_before:
        raise RuntimeError(
            "guarded nut-grasp update changed visual/local/mapping output: "
            f"before={preserved_before}, after={preserved_after}"
        )
    protected_after = {
        CONTRACT_RELATIVE_PATH.as_posix(): _sha256(
            WORKSPACE_ROOT / CONTRACT_RELATIVE_PATH
        ),
        **{
            relative: _sha256(WORKSPACE_ROOT / relative)
            for relative in verified_inputs
        },
    }
    expected_protected = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        **dict(verified_inputs),
    }
    if protected_after != expected_protected:
        raise RuntimeError("authoritative inputs changed during nut-grasp update")

    result = {
        "schema_version": "kcg_dyn_b2_b4_nut_grasp_build_result_v1",
        "task_id": "DYN-B2-B4-THREE-FINGER-GRASP",
        "generated_at_utc": _utc_now(),
        "outcome": "PASS",
        "classification": "AUTHORIZED_COUPLING_NUT_GRASP_COLLISION_GENERATED",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "assembly_control": {
            "path": _repo_relative(assembly_path),
            "sha256_before": EXPECTED_PRE_GRASP_ASSEMBLY_SHA256,
            "sha256_after": new_sha,
            "backup_path": NUT_GRASP_BACKUP_RELATIVE_PATH.as_posix(),
            "backup_sha256": _sha256(backup_path),
        },
        "grasp_collision": {
            "prim_suffix": "/LoosePlug/CouplingNut/CouplingNutGraspCollision",
            "collision_role": "coupling_nut_grasp_collision",
            "authorized_local_z_interval_m": list(
                AUTHORIZED_NUT_GRASP_LOCAL_Z_INTERVAL_M
            ),
            "authorized_outer_radius_m": AUTHORIZED_NUT_GRASP_OUTER_RADIUS_M,
            "solid_core_internal_contact_allowed": False,
            "runtime_filter_required_before_physics": True,
            "runtime_filter_owner": "d38999_tabletop_pick_smoke.py",
        },
        "preserved_outputs_before": preserved_before,
        "preserved_outputs_after": preserved_after,
        "protected_authoritative_inputs_after": protected_after,
        "visual_model_modified": False,
        "local_contact_reference_modified": False,
        "master_contract_modified": False,
        "new_backshell_created": False,
        "new_connector_geometry_candidate_created": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
        "simulation_started": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _guarded_v2_authorized_update(
    *,
    arguments: argparse.Namespace,
    output_root: Path,
    result_path: Path,
    documents: Mapping[str, str],
    contract_sha: str,
    generator_sha: str,
    verified_inputs: Mapping[str, str],
    authorized_overrides: Mapping[str, Any],
    authorized_overrides_sha256: str,
    deterministic_assembly_sha256: str,
) -> int:
    state_path = WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mismatches: dict[str, Any] = {}
    required_top_level = {
        "task_id": "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2",
        "phase": "DYN_A1_EVENT_ONSET_CALIBRATION_V2",
    }
    for key, expected in required_top_level.items():
        if state.get(key) != expected:
            mismatches[key] = {"actual": state.get(key), "expected": expected}
    allowed_statuses = {
        "INVESTIGATING",
        "ROOTCAUSE_IDENTIFIED",
        "TARGETED_FIX_IN_PROGRESS",
        "VALIDATING",
    }
    if state.get("status") not in allowed_statuses:
        mismatches["status"] = {
            "actual": state.get("status"),
            "expected_one_of": sorted(allowed_statuses),
        }
    v2_state = state.get("autonomous_dynamic_closeout_v2", {})
    if v2_state.get("current_node") != "DYN-A1-EVENT-ONSET-CALIBRATION-V2":
        mismatches["autonomous_dynamic_closeout_v2.current_node"] = {
            "actual": v2_state.get("current_node"),
            "expected": "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        }
    if v2_state.get("only_stop_status") != "PARKED_EXTERNAL":
        mismatches["autonomous_dynamic_closeout_v2.only_stop_status"] = {
            "actual": v2_state.get("only_stop_status"),
            "expected": "PARKED_EXTERNAL",
        }
    if v2_state.get("frozen_representation_sha256") != {
        "visual_complete": EXPECTED_PRESERVED_OUTPUT_SHA256[
            "D38999_VISUAL_COMPLETE_V1.usda"
        ],
        "local_contact_reference": EXPECTED_PRESERVED_OUTPUT_SHA256[
            "D38999_LOCAL_CONTACT_REFERENCE_V1.usda"
        ],
        "model_mapping": EXPECTED_PRESERVED_OUTPUT_SHA256["MODEL_MAPPING.json"],
    }:
        mismatches["autonomous_dynamic_closeout_v2.frozen_representation_sha256"] = {
            "actual": v2_state.get("frozen_representation_sha256"),
            "expected": EXPECTED_PRESERVED_OUTPUT_SHA256,
        }
    if mismatches:
        raise PermissionError(f"V2 unified generation state guard failed: {mismatches}")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    if result_path.exists():
        raise FileExistsError(f"V2 asset build result already exists: {result_path}")

    assembly_name = "D38999_ASSEMBLY_CONTROL_V1.usda"
    assembly_path = output_root / assembly_name
    actual_before = _sha256(assembly_path)
    if actual_before != EXPECTED_PRE_V2_ASSEMBLY_SHA256:
        raise PermissionError(
            "pre-V2 assembly-control fingerprint changed: "
            f"actual={actual_before}, expected={EXPECTED_PRE_V2_ASSEMBLY_SHA256}"
        )
    preserved_before = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_before != EXPECTED_PRESERVED_OUTPUT_SHA256:
        raise PermissionError(
            "frozen visual/local/mapping fingerprints differ before V2 generation: "
            f"{preserved_before}"
        )
    backup_path = (WORKSPACE_ROOT / V2_BACKUP_RELATIVE_PATH).resolve()
    if backup_path.exists():
        raise FileExistsError(f"pre-V2 assembly backup already exists: {backup_path}")

    new_text = documents[assembly_name]
    candidate_sha256 = _sha256_text(new_text)
    if candidate_sha256 != deterministic_assembly_sha256:
        raise RuntimeError(
            "two in-memory generation passes differ: "
            f"first={candidate_sha256}, second={deterministic_assembly_sha256}"
        )
    new_checks = _validate_usda_text(new_text, "D38999_ASSEMBLY_CONTROL_V1")
    required_tokens = {
        "custom int kcg:eventOnsetProxyVersion = 2": 1,
        'custom bool kcg:initialNutGraspCollisionEnabled = 1': 1,
        'custom bool kcg:newBackshellEnabled = 0': 1,
        'custom string kcg:collisionRole = "coupling_nut_grasp_collision"': 1,
        "custom double2 kcg:authorizedLocalZIntervalM = (0.009, 0.029)": 1,
        "custom double kcg:authorizedOuterRadiusM = 0.024": 1,
        "custom bool kcg:solidCoreMayContactBodyAssembly = 0": 1,
    }
    token_counts = {token: new_text.count(token) for token in required_tokens}
    if token_counts != required_tokens:
        raise ValueError(
            "V2 combined proxy tokens differ: "
            f"actual={token_counts}, expected={required_tokens}"
        )
    plan = {
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        "hypothesis_id": "A1-V2-H0-UNIFIED-DETERMINISTIC-ASSET",
        "dry_run": bool(arguments.dry_run),
        "single_asset_write": _repo_relative(assembly_path),
        "backup_write": V2_BACKUP_RELATIVE_PATH.as_posix(),
        "result_write": V2_RESULT_RELATIVE_PATH.as_posix(),
        "authorized_overrides": {
            "path": AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(),
            "sha256": authorized_overrides_sha256,
        },
        "candidate_assembly_sha256": candidate_sha256,
        "two_in_memory_generations_identical": True,
        "preserved_byte_identical": sorted(EXPECTED_PRESERVED_OUTPUT_SHA256),
        "validation": new_checks,
        "token_counts": token_counts,
        "simulation_will_start": False,
        "dynamic_pass_claimed": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assembly_path, backup_path)
    backup_sha256 = _sha256(backup_path)
    if backup_sha256 != EXPECTED_PRE_V2_ASSEMBLY_SHA256:
        raise RuntimeError("pre-V2 assembly-control backup fingerprint differs")
    _write_text_atomic(assembly_path, new_text)
    new_sha256 = _sha256(assembly_path)
    if new_sha256 != candidate_sha256:
        raise RuntimeError(
            "written V2 assembly-control fingerprint differs from candidate: "
            f"written={new_sha256}, candidate={candidate_sha256}"
        )
    preserved_after = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_after != preserved_before:
        raise RuntimeError(
            "V2 generation changed visual/local/mapping output: "
            f"before={preserved_before}, after={preserved_after}"
        )
    protected_expected = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(): (
            authorized_overrides_sha256
        ),
        **dict(verified_inputs),
    }
    protected_after = {
        relative: _sha256(WORKSPACE_ROOT / relative)
        for relative in protected_expected
    }
    if protected_after != protected_expected:
        raise RuntimeError(
            "authoritative inputs changed during V2 generation: "
            f"expected={protected_expected}, actual={protected_after}"
        )

    result = {
        "schema_version": "kcg_d38999_v2_unified_asset_build_result_v1",
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        "hypothesis_id": "A1-V2-H0-UNIFIED-DETERMINISTIC-ASSET",
        "generated_at_utc": _utc_now(),
        "outcome": "STATIC_PASS",
        "classification": "V2_UNIFIED_ASSEMBLY_CONTROL_ASSET_GENERATED",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
        },
        "authorized_overrides": {
            "path": AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(),
            "sha256": authorized_overrides_sha256,
            "validated_values": dict(authorized_overrides),
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "assembly_control": {
            "path": _repo_relative(assembly_path),
            "sha256_before": actual_before,
            "sha256_after": new_sha256,
            "backup_path": V2_BACKUP_RELATIVE_PATH.as_posix(),
            "backup_sha256": backup_sha256,
        },
        "determinism": {
            "generation_pass_1_sha256": candidate_sha256,
            "generation_pass_2_sha256": deterministic_assembly_sha256,
            "identical": True,
        },
        "token_counts": token_counts,
        "preserved_outputs_before": preserved_before,
        "preserved_outputs_after": preserved_after,
        "protected_authoritative_inputs_after": protected_after,
        "visual_model_modified": False,
        "local_contact_reference_modified": False,
        "model_mapping_modified": False,
        "master_contract_modified": False,
        "new_backshell_created": False,
        "new_connector_geometry_candidate_created": False,
        "simulation_started": False,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _remove_fine_offset_only_changes(
    text: str, *, current_generator_sha256: str
) -> str:
    offset_attribute_lines = {
        "float physxCollision:contactOffset = 1e-05",
        "float physxCollision:restOffset = 0",
    }
    normalized = "".join(
        line
        for line in text.splitlines(keepends=True)
        if line.strip() not in offset_attribute_lines
    ).replace(
        ', "PhysxCollisionAPI"',
        "",
    )
    return normalized.replace(
        current_generator_sha256,
        EXPECTED_PRE_FINE_OFFSET_GENERATOR_SHA256,
    )


def _guarded_v2_fine_collision_offset_fix(
    *,
    arguments: argparse.Namespace,
    output_root: Path,
    result_path: Path,
    documents: Mapping[str, str],
    contract_sha: str,
    generator_sha: str,
    verified_inputs: Mapping[str, str],
    authorized_overrides: Mapping[str, Any],
    authorized_overrides_sha256: str,
    deterministic_assembly_sha256: str,
) -> int:
    state_path = WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    v2_state = state.get("autonomous_dynamic_closeout_v2", {})
    audit_state = v2_state.get("collision_audit", {})
    required = {
        "task_id": (state.get("task_id"), "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2"),
        "phase": (state.get("phase"), "DYN_A1_EVENT_ONSET_CALIBRATION_V2"),
        "status": (state.get("status"), "TARGETED_FIX_IN_PROGRESS"),
        "current_node": (
            v2_state.get("current_node"),
            "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        ),
        "collision_audit.status": (audit_state.get("status"), "ROOTCAUSE_IDENTIFIED"),
        "collision_audit.classification": (
            audit_state.get("classification"),
            "COOKED_GEOMETRY_SUPPORTED_EXPLICIT_OFFSETS_MISSING",
        ),
        "collision_audit.target_collider_count": (
            audit_state.get("target_collider_count"),
            269,
        ),
        "collision_audit.explicit_offset_collider_count": (
            audit_state.get("explicit_offset_collider_count"),
            0,
        ),
        "collision_audit.maximum_cooked_world_bound_error_m": (
            audit_state.get("maximum_cooked_world_bound_error_m"),
            0.0,
        ),
        "collision_audit.adjusted_thickness_warning_count": (
            audit_state.get("adjusted_thickness_warning_count"),
            0,
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if mismatches:
        raise PermissionError(f"V2 fine-offset state guard failed: {mismatches}")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    if result_path.exists():
        raise FileExistsError(f"V2 fine-offset result already exists: {result_path}")

    assembly_name = "D38999_ASSEMBLY_CONTROL_V1.usda"
    assembly_path = output_root / assembly_name
    actual_before = _sha256(assembly_path)
    if actual_before != EXPECTED_PRE_FINE_OFFSET_ASSEMBLY_SHA256:
        raise PermissionError(
            "pre-fine-offset assembly fingerprint changed: "
            f"actual={actual_before}, expected={EXPECTED_PRE_FINE_OFFSET_ASSEMBLY_SHA256}"
        )
    original_v2_result = WORKSPACE_ROOT / V2_RESULT_RELATIVE_PATH
    if _sha256(original_v2_result) != EXPECTED_PRE_FINE_OFFSET_RESULT_SHA256:
        raise PermissionError("original V2 unified build result fingerprint changed")
    original_v2_backup = WORKSPACE_ROOT / V2_BACKUP_RELATIVE_PATH
    if _sha256(original_v2_backup) != EXPECTED_PRE_V2_ASSEMBLY_SHA256:
        raise PermissionError("original pre-V2 assembly backup fingerprint changed")
    preserved_before = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_before != EXPECTED_PRESERVED_OUTPUT_SHA256:
        raise PermissionError(
            "frozen visual/local/mapping fingerprints differ before fine-offset fix"
        )
    backup_path = (WORKSPACE_ROOT / V2_FINE_OFFSET_BACKUP_RELATIVE_PATH).resolve()
    if backup_path.exists():
        raise FileExistsError(f"pre-fine-offset backup already exists: {backup_path}")

    new_text = documents[assembly_name]
    candidate_sha256 = _sha256_text(new_text)
    if candidate_sha256 != deterministic_assembly_sha256:
        raise RuntimeError("two fine-offset generation passes differ")
    current_text = assembly_path.read_text(encoding="utf-8")
    normalized_new_text = _remove_fine_offset_only_changes(
        new_text,
        current_generator_sha256=generator_sha,
    )
    if normalized_new_text != current_text:
        raise RuntimeError(
            "fine-offset candidate changes content beyond API/offset authoring and generator SHA"
        )
    new_checks = _validate_usda_text(new_text, "D38999_ASSEMBLY_CONTROL_V1")
    required_token_counts = {
        "PhysxCollisionAPI": EXPECTED_ACTIVE_FINE_COLLIDER_COUNT,
        "float physxCollision:contactOffset = 1e-05": (
            EXPECTED_ACTIVE_FINE_COLLIDER_COUNT
        ),
        "float physxCollision:restOffset = 0": EXPECTED_ACTIVE_FINE_COLLIDER_COUNT,
    }
    token_counts = {
        token: new_text.count(token) for token in required_token_counts
    }
    if token_counts != required_token_counts:
        raise ValueError(
            "fine-connector explicit offset counts differ: "
            f"actual={token_counts}, expected={required_token_counts}"
        )
    plan = {
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        "hypothesis_id": "A1-V2-H1-COOKED-SURFACE-AND-CONTACT-MARGIN-ONSET",
        "dry_run": bool(arguments.dry_run),
        "single_asset_write": _repo_relative(assembly_path),
        "backup_write": V2_FINE_OFFSET_BACKUP_RELATIVE_PATH.as_posix(),
        "result_write": V2_FINE_OFFSET_RESULT_RELATIVE_PATH.as_posix(),
        "candidate_assembly_sha256": candidate_sha256,
        "two_in_memory_generations_identical": True,
        "geometry_and_non_offset_text_byte_identical": True,
        "token_counts": token_counts,
        "preserved_byte_identical": sorted(EXPECTED_PRESERVED_OUTPUT_SHA256),
        "simulation_will_start": False,
        "dynamic_pass_claimed": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assembly_path, backup_path)
    if _sha256(backup_path) != EXPECTED_PRE_FINE_OFFSET_ASSEMBLY_SHA256:
        raise RuntimeError("pre-fine-offset assembly backup fingerprint differs")
    _write_text_atomic(assembly_path, new_text)
    new_sha256 = _sha256(assembly_path)
    if new_sha256 != candidate_sha256:
        raise RuntimeError("written fine-offset asset differs from deterministic candidate")
    preserved_after = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_after != preserved_before:
        raise RuntimeError("fine-offset fix changed frozen visual/local/mapping output")
    protected_expected = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(): (
            authorized_overrides_sha256
        ),
        **dict(verified_inputs),
    }
    protected_after = {
        relative: _sha256(WORKSPACE_ROOT / relative)
        for relative in protected_expected
    }
    if protected_after != protected_expected:
        raise RuntimeError("authoritative inputs changed during fine-offset fix")

    result = {
        "schema_version": "kcg_d38999_v2_fine_collision_offset_fix_result_v1",
        "task_id": "DYN-A1-EVENT-ONSET-CALIBRATION-V2",
        "hypothesis_id": "A1-V2-H1-COOKED-SURFACE-AND-CONTACT-MARGIN-ONSET",
        "generated_at_utc": _utc_now(),
        "outcome": "STATIC_PASS",
        "classification": "EXPLICIT_FINE_COLLISION_OFFSETS_AUTHORED",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
        },
        "authorized_overrides": {
            "path": AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(),
            "sha256": authorized_overrides_sha256,
            "validated_values": dict(authorized_overrides),
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "assembly_control": {
            "path": _repo_relative(assembly_path),
            "sha256_before": actual_before,
            "sha256_after": new_sha256,
            "backup_path": V2_FINE_OFFSET_BACKUP_RELATIVE_PATH.as_posix(),
            "backup_sha256": _sha256(backup_path),
        },
        "determinism": {
            "generation_pass_1_sha256": candidate_sha256,
            "generation_pass_2_sha256": deterministic_assembly_sha256,
            "identical": True,
        },
        "explicit_offset_authoring": {
            "active_fine_collider_count": EXPECTED_ACTIVE_FINE_COLLIDER_COUNT,
            "contact_offset_m": FINE_CONNECTOR_CONTACT_OFFSET_M,
            "rest_offset_m": FINE_CONNECTOR_REST_OFFSET_M,
            "token_counts": token_counts,
            "geometry_and_non_offset_text_byte_identical": True,
            "min_thickness_modified": False,
        },
        "prior_unified_build_result": {
            "path": V2_RESULT_RELATIVE_PATH.as_posix(),
            "sha256": EXPECTED_PRE_FINE_OFFSET_RESULT_SHA256,
        },
        "preserved_outputs_before": preserved_before,
        "preserved_outputs_after": preserved_after,
        "protected_authoritative_inputs_after": protected_after,
        "visual_model_modified": False,
        "local_contact_reference_modified": False,
        "model_mapping_modified": False,
        "master_contract_modified": False,
        "geometry_modified": False,
        "simulation_started": False,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _guarded_v2_nut_body_shoulder_repair(
    *,
    arguments: argparse.Namespace,
    output_root: Path,
    result_path: Path,
    documents: Mapping[str, str],
    contract_sha: str,
    physical_contract_sha: str,
    generator_sha: str,
    verified_inputs: Mapping[str, str],
    authorized_overrides: Mapping[str, Any],
    authorized_overrides_sha256: str,
    deterministic_assembly_sha256: str,
) -> int:
    state_path = WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    v2_state = state.get("autonomous_dynamic_closeout_v2", {})
    nominal = v2_state.get("nominal_insertion", {})
    blocker = state.get("current_blocker", {})
    required = {
        "task_id": (state.get("task_id"), "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2"),
        "phase": (state.get("phase"), "DYN_A2_NOMINAL_INSERTION_V2"),
        "status": (state.get("status"), "ROOTCAUSE_IDENTIFIED"),
        "current_node": (v2_state.get("current_node"), "DYN-A2-NOMINAL-INSERTION-V2"),
        "nominal.status": (nominal.get("status"), "ROOTCAUSE_IDENTIFIED"),
        "nominal.hypothesis_id": (
            nominal.get("hypothesis_id"),
            "A2-V2-H6-MISSING-NUT-BODY-PHYSICAL-SHOULDER-LOAD-PATH",
        ),
        "blocker.classification": (
            blocker.get("classification"),
            "MISSING_AUTHORIZED_NUT_BODY_PHYSICAL_SHOULDER_LOAD_PATH",
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if mismatches:
        raise PermissionError(f"A2 H6 shoulder state guard failed: {mismatches}")
    rootcause_path = WORKSPACE_ROOT / V2_SHOULDER_ROOTCAUSE_RELATIVE_PATH
    if _sha256(rootcause_path) != EXPECTED_SHOULDER_ROOTCAUSE_SHA256:
        raise PermissionError("A2 H6 shoulder root-cause evidence fingerprint changed")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    if result_path.exists():
        raise FileExistsError(f"A2 H6 shoulder result already exists: {result_path}")

    assembly_name = "D38999_ASSEMBLY_CONTROL_V1.usda"
    assembly_path = output_root / assembly_name
    actual_before = _sha256(assembly_path)
    if actual_before != EXPECTED_PRE_SHOULDER_ASSEMBLY_SHA256:
        raise PermissionError(
            "pre-shoulder assembly fingerprint changed: "
            f"actual={actual_before}, expected={EXPECTED_PRE_SHOULDER_ASSEMBLY_SHA256}"
        )
    current_text = assembly_path.read_text(encoding="utf-8")
    forbidden_before = {
        'def Xform "NutBearingShoulders"': 0,
        'custom string kcg:collisionRole = "hard_nut_body_shoulder"': 0,
    }
    actual_before_counts = {
        token: current_text.count(token) for token in forbidden_before
    }
    if actual_before_counts != forbidden_before:
        raise PermissionError(
            f"pre-shoulder asset unexpectedly contains shoulder geometry: {actual_before_counts}"
        )
    preserved_before = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_before != EXPECTED_PRESERVED_OUTPUT_SHA256:
        raise PermissionError("frozen visual/local/mapping fingerprints differ before H6 repair")

    new_text = documents[assembly_name]
    candidate_sha256 = _sha256_text(new_text)
    if candidate_sha256 != deterministic_assembly_sha256:
        raise RuntimeError("two A2 H6 shoulder generation passes differ")
    required_token_counts = {
        "custom bool kcg:nutBodyPhysicalShoulderEnabled = 1": 1,
        "custom string kcg:physicalShoulderSourceSha256 = "
        + _usd_string(physical_contract_sha): 1,
        'def Xform "NutBearingShoulders"': 2,
        'def Xform "PositiveStop"': 2,
        'def Xform "NegativeStop"': 2,
        'def Cylinder "AnalyticCap"': 2,
        'def Sphere "AnalyticSphere_': 6,
        'custom string kcg:collisionRole = "hard_nut_body_shoulder"': 8,
        "PhysxCollisionAPI": EXPECTED_POST_SHOULDER_ACTIVE_FINE_COLLIDER_COUNT,
        "float physxCollision:contactOffset = 1e-05": (
            EXPECTED_POST_SHOULDER_ACTIVE_FINE_COLLIDER_COUNT
        ),
        "float physxCollision:restOffset = 0": (
            EXPECTED_POST_SHOULDER_ACTIVE_FINE_COLLIDER_COUNT
        ),
    }
    token_counts = {token: new_text.count(token) for token in required_token_counts}
    if token_counts != required_token_counts:
        raise ValueError(
            "A2 H6 shoulder token counts differ: "
            f"actual={token_counts}, expected={required_token_counts}"
        )
    plan = {
        "task_id": "DYN-A2-NOMINAL-INSERTION-V2",
        "hypothesis_id": "A2-V2-H6-MISSING-NUT-BODY-PHYSICAL-SHOULDER-LOAD-PATH",
        "dry_run": bool(arguments.dry_run),
        "single_asset_write": _repo_relative(assembly_path),
        "backup_write": V2_SHOULDER_BACKUP_RELATIVE_PATH.as_posix(),
        "result_write": V2_SHOULDER_RESULT_RELATIVE_PATH.as_posix(),
        "candidate_assembly_sha256": candidate_sha256,
        "two_in_memory_generations_identical": True,
        "shoulder_leaf_collider_count": 8,
        "preserved_byte_identical": sorted(EXPECTED_PRESERVED_OUTPUT_SHA256),
        "token_counts": token_counts,
        "simulation_will_start": False,
        "dynamic_pass_claimed": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backup_path = (WORKSPACE_ROOT / V2_SHOULDER_BACKUP_RELATIVE_PATH).resolve()
    if backup_path.exists():
        raise FileExistsError(f"pre-H6 assembly backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assembly_path, backup_path)
    if _sha256(backup_path) != EXPECTED_PRE_SHOULDER_ASSEMBLY_SHA256:
        raise RuntimeError("pre-H6 assembly backup fingerprint differs")
    _write_text_atomic(assembly_path, new_text)
    new_sha256 = _sha256(assembly_path)
    if new_sha256 != candidate_sha256:
        raise RuntimeError("written H6 shoulder asset differs from deterministic candidate")
    preserved_after = {
        name: _sha256(output_root / name)
        for name in EXPECTED_PRESERVED_OUTPUT_SHA256
    }
    if preserved_after != preserved_before:
        raise RuntimeError("H6 shoulder repair changed frozen visual/local/mapping output")
    protected_expected = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        PHYSICAL_CONTRACT_RELATIVE_PATH.as_posix(): physical_contract_sha,
        AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(): authorized_overrides_sha256,
        **dict(verified_inputs),
    }
    protected_after = {
        relative: _sha256(WORKSPACE_ROOT / relative)
        for relative in protected_expected
    }
    if protected_after != protected_expected:
        raise RuntimeError("authoritative inputs changed during H6 shoulder repair")

    result = {
        "schema_version": "kcg_d38999_a2_v2_shoulder_targeted_fix_v1",
        "task_id": "DYN-A2-NOMINAL-INSERTION-V2",
        "hypothesis_id": "A2-V2-H6-MISSING-NUT-BODY-PHYSICAL-SHOULDER-LOAD-PATH",
        "generated_at_utc": _utc_now(),
        "outcome": "STATIC_PASS_AWAITING_DYNAMIC_VALIDATION",
        "classification": "FROZEN_NUT_BODY_PHYSICAL_SHOULDER_RESTORED",
        "rootcause_evidence": {
            "path": V2_SHOULDER_ROOTCAUSE_RELATIVE_PATH.as_posix(),
            "sha256": EXPECTED_SHOULDER_ROOTCAUSE_SHA256,
        },
        "master_contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
            "modified": False,
        },
        "physical_contract": {
            "path": PHYSICAL_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": physical_contract_sha,
            "modified": False,
        },
        "authorized_overrides": {
            "path": AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH.as_posix(),
            "sha256": authorized_overrides_sha256,
            "modified": False,
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "assembly_control": {
            "path": _repo_relative(assembly_path),
            "sha256_before": actual_before,
            "sha256_after": new_sha256,
            "backup_path": V2_SHOULDER_BACKUP_RELATIVE_PATH.as_posix(),
            "backup_sha256": _sha256(backup_path),
        },
        "shoulder_geometry": {
            "body0_analytic_cap_count": 2,
            "body1_analytic_sphere_count": 6,
            "hard_shoulder_leaf_collider_count": 8,
            "physical_endplay_m": {"low": -0.00005, "high": 0.00005},
            "joint_backup_limits_m": {"low": -0.00015, "high": 0.00015},
            "load_path": "thread_to_nut_to_physical_shoulder_to_plug_body",
            "source": "frozen_physical_contract_existing_recipe",
        },
        "determinism": {
            "generation_pass_1_sha256": candidate_sha256,
            "generation_pass_2_sha256": deterministic_assembly_sha256,
            "identical": True,
        },
        "token_counts": token_counts,
        "preserved_outputs_before": preserved_before,
        "preserved_outputs_after": preserved_after,
        "protected_authoritative_inputs_after": protected_after,
        "controller_modified": False,
        "visual_model_modified": False,
        "local_contact_reference_modified": False,
        "model_mapping_modified": False,
        "new_connector_geometry_candidate_created": False,
        "event_positions_modified": False,
        "safety_limits_modified": False,
        "simulation_started": False,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    guarded_mode_count = sum(
        bool(value)
        for value in (
            arguments.event_onset_calibration,
            arguments.initial_nut_grasp_collision,
            arguments.authorized_overrides_v2,
            arguments.v2_fine_collision_offset_fix,
            arguments.v2_nut_body_shoulder_repair,
        )
    )
    if guarded_mode_count > 1:
        raise ValueError("guarded update modes are mutually exclusive")
    contract_path = _require_exact_path(
        arguments.contract, CONTRACT_RELATIVE_PATH, "master contract"
    )
    output_root = _require_exact_path(
        arguments.output_root, OUTPUT_ROOT_RELATIVE_PATH, "output root"
    )
    if arguments.v2_nut_body_shoulder_repair:
        result_relative = V2_SHOULDER_RESULT_RELATIVE_PATH
    elif arguments.v2_fine_collision_offset_fix:
        result_relative = V2_FINE_OFFSET_RESULT_RELATIVE_PATH
    elif arguments.authorized_overrides_v2:
        result_relative = V2_RESULT_RELATIVE_PATH
    elif arguments.event_onset_calibration:
        result_relative = EVENT_ONSET_RESULT_RELATIVE_PATH
    elif arguments.initial_nut_grasp_collision:
        result_relative = NUT_GRASP_RESULT_RELATIVE_PATH
    else:
        result_relative = RESULT_RELATIVE_PATH
    result_path = _require_exact_path(arguments.result, result_relative, "result")
    document = _load_and_validate_contract(contract_path)
    physical_contract_path = _require_exact_path(
        str(WORKSPACE_ROOT / PHYSICAL_CONTRACT_RELATIVE_PATH),
        PHYSICAL_CONTRACT_RELATIVE_PATH,
        "physical contract",
    )
    physical_shoulder_recipe = _load_and_validate_physical_shoulder_recipe(
        physical_contract_path, document
    )
    verified_inputs = _validate_authoritative_inputs(document)
    generator_path = (WORKSPACE_ROOT / GENERATOR_RELATIVE_PATH).resolve()
    generator_sha = _sha256(generator_path)
    contract_sha = _sha256(contract_path)
    physical_contract_sha = _sha256(physical_contract_path)
    authorized_overrides: dict[str, Any] | None = None
    authorized_overrides_sha256: str | None = None
    if (
        arguments.authorized_overrides_v2
        or arguments.v2_fine_collision_offset_fix
        or arguments.v2_nut_body_shoulder_repair
    ):
        authorized_overrides_path = _require_exact_path(
            arguments.authorized_overrides,
            AUTHORIZED_OVERRIDES_V2_RELATIVE_PATH,
            "V2 authorized overrides",
        )
        authorized_overrides = _load_and_validate_authorized_overrides_v2(
            authorized_overrides_path
        )
        authorized_overrides_sha256 = _sha256(authorized_overrides_path)
    documents, mapping = _build_documents(
        document,
        output_root=output_root,
        contract_sha=contract_sha,
        physical_contract_sha=physical_contract_sha,
        physical_shoulder_recipe=physical_shoulder_recipe,
        generator_sha=generator_sha,
        include_coupling_nut_grasp_collision=(
            arguments.initial_nut_grasp_collision
        ),
        authorized_overrides_v2=authorized_overrides,
        authorized_overrides_v2_sha256=authorized_overrides_sha256,
    )
    if (
        arguments.authorized_overrides_v2
        or arguments.v2_fine_collision_offset_fix
        or arguments.v2_nut_body_shoulder_repair
    ):
        second_documents, second_mapping = _build_documents(
            document,
            output_root=output_root,
            contract_sha=contract_sha,
            physical_contract_sha=physical_contract_sha,
            physical_shoulder_recipe=physical_shoulder_recipe,
            generator_sha=generator_sha,
            authorized_overrides_v2=authorized_overrides,
            authorized_overrides_v2_sha256=authorized_overrides_sha256,
        )
        if documents != second_documents or mapping != second_mapping:
            raise RuntimeError("two V2 generation passes are not deterministic")
        if arguments.v2_nut_body_shoulder_repair:
            return _guarded_v2_nut_body_shoulder_repair(
                arguments=arguments,
                output_root=output_root,
                result_path=result_path,
                documents=documents,
                contract_sha=contract_sha,
                physical_contract_sha=physical_contract_sha,
                generator_sha=generator_sha,
                verified_inputs=verified_inputs,
                authorized_overrides=authorized_overrides,
                authorized_overrides_sha256=authorized_overrides_sha256,
                deterministic_assembly_sha256=_sha256_text(
                    second_documents["D38999_ASSEMBLY_CONTROL_V1.usda"]
                ),
            )
        if arguments.v2_fine_collision_offset_fix:
            return _guarded_v2_fine_collision_offset_fix(
                arguments=arguments,
                output_root=output_root,
                result_path=result_path,
                documents=documents,
                contract_sha=contract_sha,
                generator_sha=generator_sha,
                verified_inputs=verified_inputs,
                authorized_overrides=authorized_overrides,
                authorized_overrides_sha256=authorized_overrides_sha256,
                deterministic_assembly_sha256=_sha256_text(
                    second_documents["D38999_ASSEMBLY_CONTROL_V1.usda"]
                ),
            )
        return _guarded_v2_authorized_update(
            arguments=arguments,
            output_root=output_root,
            result_path=result_path,
            documents=documents,
            contract_sha=contract_sha,
            generator_sha=generator_sha,
            verified_inputs=verified_inputs,
            authorized_overrides=authorized_overrides,
            authorized_overrides_sha256=authorized_overrides_sha256,
            deterministic_assembly_sha256=_sha256_text(
                second_documents["D38999_ASSEMBLY_CONTROL_V1.usda"]
            ),
        )
    if arguments.event_onset_calibration:
        return _guarded_event_onset_update(
            arguments=arguments,
            document=document,
            output_root=output_root,
            result_path=result_path,
            documents=documents,
            contract_sha=contract_sha,
            generator_sha=generator_sha,
            verified_inputs=verified_inputs,
        )
    if arguments.initial_nut_grasp_collision:
        return _guarded_nut_grasp_update(
            arguments=arguments,
            output_root=output_root,
            result_path=result_path,
            documents=documents,
            contract_sha=contract_sha,
            generator_sha=generator_sha,
            verified_inputs=verified_inputs,
        )
    planned_files = [
        (OUTPUT_ROOT_RELATIVE_PATH / filename).as_posix()
        for filename in sorted(documents)
    ]
    planned_files.append((OUTPUT_ROOT_RELATIVE_PATH / "MODEL_MAPPING.json").as_posix())
    planned_files.append(RESULT_RELATIVE_PATH.as_posix())
    plan = {
        "task_id": "TASK-R12-MULTILAYER-003",
        "dry_run": bool(arguments.dry_run),
        "contract_sha256": contract_sha,
        "generator_sha256": generator_sha,
        "authoritative_inputs_verified": verified_inputs,
        "planned_writes": planned_files,
        "simulation_will_start": False,
        "existing_file_overwrite_allowed": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite build result: {result_path}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = {
        CONTRACT_RELATIVE_PATH.as_posix(): contract_sha,
        **verified_inputs,
    }
    with tempfile.TemporaryDirectory(
        prefix=".d38999_multilayer_v1.", dir=output_root.parent
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        for filename, text in documents.items():
            (temporary_root / filename).write_text(text, encoding="utf-8")
        mapping_path = temporary_root / "MODEL_MAPPING.json"
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated_hashes = {
            filename: _sha256(temporary_root / filename)
            for filename in sorted([*documents, "MODEL_MAPPING.json"])
        }
        temporary_root.replace(output_root)

    protected_after = {
        relative: _sha256(WORKSPACE_ROOT / relative) for relative in protected_before
    }
    protected_changed = sorted(
        relative
        for relative, before_sha in protected_before.items()
        if protected_after[relative] != before_sha
    )
    if protected_changed:
        raise RuntimeError(f"protected inputs changed during generation: {protected_changed}")
    result = {
        "schema_version": "kcg_task_r12_multilayer_003_build_result_v1",
        "task_id": "TASK-R12-MULTILAYER-003",
        "generated_at_utc": _utc_now(),
        "outcome": "PASS",
        "classification": "THREE_REPRESENTATIONS_GENERATED_FROM_ONE_CONTRACT",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
        },
        "generator": {
            "path": GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": generator_sha,
        },
        "output_root": OUTPUT_ROOT_RELATIVE_PATH.as_posix(),
        "generated_files_sha256": generated_hashes,
        "representation_names": list(REPRESENTATIONS),
        "traceable_pair_count": 61,
        "cross_label_effect_count": 0,
        "segmented_socket_entry_micro_wedge_count": 0,
        "segmented_pin_barrier_micro_wedge_count": 0,
        "all_combination_micro_contact_count": 0,
        "visual_high_detail_reference": mapping["representations"][
            "D38999_VISUAL_COMPLETE_V1"
        ]["direct_reference"],
        "local_high_detail_reference": mapping["representations"][
            "D38999_LOCAL_CONTACT_REFERENCE_V1"
        ]["direct_reference"],
        "protected_input_hashes_before": protected_before,
        "protected_input_hashes_after": protected_after,
        "protected_inputs_changed": protected_changed,
        "simulation_started": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    _write_text_atomic(
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
