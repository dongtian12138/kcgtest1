#!/usr/bin/env python3

"""Independently validate the frozen D38999 multilayer representations.

The validator does not import the generator and does not start Isaac Sim.  It
reads the master contract, generated mapping, build result, and USDA text, then
publishes one auditable static-consistency result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
)
BUILD_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/BUILD_RESULT.json"
)
OUTPUT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-004/STATIC_CONSISTENCY.json"
)
MASTER_STATE_RELATIVE_PATH = Path("artifacts/agent_control/MASTER_STATE.json")
EXPECTED_CONTRACT_SHA256 = (
    "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
)
EXPECTED_BUILD_RESULT_SHA256 = (
    "f73a7734008c2cab54c92a05cc0744a2eecc04516592b07da87f5511c96ec113"
)
EXPECTED_GENERATOR_SHA256 = (
    "8da5aa34609dc9b5d49f16c82d61fb94a63f4800f7dee04acaf4e8da8782bff3"
)
TOLERANCE = 1.0e-12


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate D38999 multilayer static consistency")
    parser.add_argument("--phase", required=True, choices=("static",))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--build-result", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_path(requested: str, relative: Path, role: str) -> Path:
    actual = Path(requested).expanduser().resolve()
    expected = (WORKSPACE_ROOT / relative).resolve()
    if actual != expected:
        raise PermissionError(
            f"{role} path is frozen: requested={actual}, expected={expected}"
        )
    return actual


def _close(actual: float, expected: float, tolerance: float = TOLERANCE) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def _sequence_close(actual: Sequence[Any], expected: Sequence[Any]) -> bool:
    return len(actual) == len(expected) and all(
        _close(float(left), float(right)) for left, right in zip(actual, expected)
    )


def _prim_block(text: str, prim_type: str, name: str) -> str:
    pattern = re.compile(
        rf'\bdef\s+{re.escape(prim_type)}\s+"{re.escape(name)}"'
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing {prim_type} prim {name}")
    opening = text.find("{", match.end())
    if opening < 0:
        raise ValueError(f"missing opening brace for {prim_type} {name}")
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise ValueError(f"unterminated {prim_type} prim {name}")


def _number(block: str, attribute: str) -> float:
    match = re.search(
        rf"\b{re.escape(attribute)}\s*=\s*([-+0-9.eE]+)", block
    )
    if match is None:
        raise ValueError(f"missing numeric attribute {attribute}")
    return float(match.group(1))


def _vector(block: str, attribute: str) -> list[float]:
    match = re.search(
        rf"\b{re.escape(attribute)}\s*=\s*\(([^)]*)\)", block
    )
    if match is None:
        raise ValueError(f"missing vector attribute {attribute}")
    return [float(part.strip()) for part in match.group(1).split(",")]


def _mesh_points(block: str) -> list[tuple[float, float, float]]:
    match = re.search(r"point3f\[\]\s+points\s*=\s*\[(.*?)\]", block, re.DOTALL)
    if match is None:
        raise ValueError("mesh points are missing")
    points = []
    for row in re.findall(r"\(([^()]*)\)", match.group(1)):
        values = tuple(float(part.strip()) for part in row.split(","))
        if len(values) != 3:
            raise ValueError("mesh point does not have three coordinates")
        points.append(values)
    if not points:
        raise ValueError("mesh has no points")
    return points


def _ring_measurements(text: str, name: str) -> dict[str, float | int]:
    points = _mesh_points(_prim_block(text, "Mesh", name))
    radii = [math.hypot(point[0], point[1]) for point in points]
    z_values = [point[2] for point in points]
    return {
        "point_count": len(points),
        "inner_radius_m": min(radii),
        "outer_radius_m": max(radii),
        "z_min_m": min(z_values),
        "z_max_m": max(z_values),
    }


def _prim_paths(text: str) -> set[str]:
    paths: set[str] = set()
    stack: list[str] = []
    pending_name: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r'def\s+\w+\s+"([^"]+)"', line)
        if match:
            if pending_name is not None:
                raise ValueError("nested prim declaration before prior prim opened")
            pending_name = match.group(1)
        for char in line:
            if char == "{":
                if pending_name is not None:
                    stack.append(pending_name)
                    paths.add("/" + "/".join(stack))
                    pending_name = None
            elif char == "}":
                if not stack:
                    raise ValueError("USDA closing brace without an open prim")
                stack.pop()
    if stack or pending_name is not None:
        raise ValueError("USDA prim nesting is incomplete")
    return paths


def _reference_target(asset_path: Path, text: str) -> tuple[Path, str]:
    match = re.search(r"prepend\s+references\s*=\s*@([^@]+)@<([^>]+)>", text)
    if match is None:
        raise ValueError(f"direct reference missing from {asset_path}")
    target = (asset_path.parent / match.group(1)).resolve()
    return target, match.group(2)


def _mass_measurement(text: str, body_name: str) -> dict[str, Any]:
    block = _prim_block(text, "Xform", body_name)
    return {
        "mass_kg": _number(block, "physics:mass"),
        "center_of_mass_m": _vector(block, "physics:centerOfMass"),
        "diagonal_inertia_kg_m2": _vector(block, "physics:diagonalInertia"),
        "principal_axes_wxyz": _vector(block, "physics:principalAxes"),
    }


def _mass_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        _close(actual["mass_kg"], expected["mass_kg"])
        and _sequence_close(actual["center_of_mass_m"], expected["center_of_mass_m"])
        and _sequence_close(
            actual["diagonal_inertia_kg_m2"], expected["diagonal_inertia_kg_m2"]
        )
        and _sequence_close(
            actual["principal_axes_wxyz"], expected["principal_axes_wxyz"]
        )
    )


def _expected_logical_paths(
    contract: Mapping[str, Any], pair_root: str, labels: Sequence[str]
) -> set[str]:
    expected: set[str] = {pair_root}
    for name, suffix in contract["logical_path_interface"]["logical_parts"].items():
        if name == "pair_root":
            continue
        if "{label}" in suffix:
            expected.update(f"{pair_root}/{suffix.format(label=label)}" for label in labels)
        else:
            expected.add(f"{pair_root}/{suffix}")
    return expected


def _pair_relations_valid(assembly_text: str, labels: Sequence[str], pair_root: str) -> bool:
    for label in labels:
        expected_pin = f"{pair_root}/FixedReceptacle/Contacts/Pin_{label}"
        expected_socket = f"{pair_root}/LoosePlug/BodyAssembly/Contacts/Socket_{label}"
        expected_barrier = f"{pair_root}/FixedReceptacle/Contacts/Barrier_{label}"
        for effect_name in (f"SmoothEntry_{label}", f"DeepPin_{label}"):
            block = _prim_block(assembly_text, "Scope", effect_name)
            if f"rel kcg:pin = <{expected_pin}>" not in block:
                return False
            if f"rel kcg:socket = <{expected_socket}>" not in block:
                return False
            if "custom bool kcg:sameLabelOnly = 1" not in block:
                return False
        isolation = _prim_block(assembly_text, "Scope", f"PinIsolation_{label}")
        if f"rel kcg:pin = <{expected_pin}>" not in isolation:
            return False
        if f"rel kcg:socket = <{expected_socket}>" not in isolation:
            return False
        if f"rel kcg:barrier = <{expected_barrier}>" not in isolation:
            return False
        if "custom bool kcg:sameLabelOnly = 1" not in isolation:
            return False
    return True


def _write_result(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite static result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale static-result temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    contract_path = _exact_path(arguments.contract, CONTRACT_RELATIVE_PATH, "contract")
    build_result_path = _exact_path(
        arguments.build_result, BUILD_RESULT_RELATIVE_PATH, "build result"
    )
    output_path = _exact_path(arguments.output, OUTPUT_RELATIVE_PATH, "output")
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise PermissionError("master contract fingerprint changed")
    if _sha256(build_result_path) != EXPECTED_BUILD_RESULT_SHA256:
        raise PermissionError("build-result fingerprint changed")
    state = json.loads((WORKSPACE_ROOT / MASTER_STATE_RELATIVE_PATH).read_text())
    if state.get("task_id") != "TASK-R12-MULTILAYER-004":
        raise PermissionError("static validation is authorized only during TASK-R12-MULTILAYER-004")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    build = json.loads(build_result_path.read_text(encoding="utf-8"))
    output_root = (WORKSPACE_ROOT / build["output_root"]).resolve()
    mapping_path = output_root / "MODEL_MAPPING.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    visual_path = output_root / "D38999_VISUAL_COMPLETE_V1.usda"
    assembly_path = output_root / "D38999_ASSEMBLY_CONTROL_V1.usda"
    local_path = output_root / "D38999_LOCAL_CONTACT_REFERENCE_V1.usda"
    visual_text = visual_path.read_text(encoding="utf-8")
    assembly_text = assembly_path.read_text(encoding="utf-8")
    local_text = local_path.read_text(encoding="utf-8")

    actual_hashes = {
        path.name: _sha256(path)
        for path in (visual_path, assembly_path, local_path, mapping_path)
    }
    inputs_unchanged = actual_hashes == build["generated_files_sha256"]
    contract_link_equal = (
        mapping["contract"]["sha256"]
        == build["contract"]["sha256"]
        == EXPECTED_CONTRACT_SHA256
        and all(EXPECTED_CONTRACT_SHA256 in text for text in (visual_text, assembly_text, local_text))
    )
    generator_link_equal = (
        mapping["generator"]["sha256"]
        == build["generator"]["sha256"]
        == EXPECTED_GENERATOR_SHA256
        and all(EXPECTED_GENERATOR_SHA256 in text for text in (visual_text, assembly_text, local_text))
    )

    visual_target, visual_target_prim = _reference_target(visual_path, visual_text)
    local_target, local_target_prim = _reference_target(local_path, local_text)
    expected_reference = (
        WORKSPACE_ROOT
        / contract["representation_requirements"]["D38999_LOCAL_CONTACT_REFERENCE_V1"][
            "direct_reference_path"
        ]
    ).resolve()
    expected_reference_sha = contract["representation_requirements"][
        "D38999_LOCAL_CONTACT_REFERENCE_V1"
    ]["direct_reference_sha256"]
    direct_reference_equal = (
        visual_target
        == local_target
        == expected_reference
        and visual_target_prim == local_target_prim == "/World/D38999Shell25JKeyedPhysicalV3"
        and _sha256(expected_reference) == expected_reference_sha
    )

    representation = mapping["representations"]["D38999_ASSEMBLY_CONTROL_V1"]
    pair_root = representation["pair_root"]
    pairs = representation["pair_effects"]
    labels = [str(row["label"]) for row in contract["contact_layout"]["pairs"]]
    pair_labels = [str(row["label"]) for row in pairs]
    scale = float(contract["contact_layout"]["coordinate_scale_m_per_in"])
    pin_center_checks: list[bool] = []
    socket_center_checks: list[bool] = []
    mapped_center_checks: list[bool] = []
    for expected, mapped in zip(contract["contact_layout"]["pairs"], pairs):
        label = str(expected["label"])
        expected_xy = [float(value) * scale for value in expected["center_in"]]
        pin_block = _prim_block(assembly_text, "Cylinder", f"Pin_{label}")
        socket_block = _prim_block(assembly_text, "Xform", f"Socket_{label}")
        pin_xy = _vector(pin_block, "xformOp:translate")[:2]
        socket_xy = _vector(socket_block, "kcg:centerM")
        pin_center_checks.append(_sequence_close(pin_xy, expected_xy))
        socket_center_checks.append(_sequence_close(socket_xy, expected_xy))
        mapped_center_checks.append(
            mapped["label"] == label
            and _sequence_close(mapped["center_in"], expected["center_in"])
            and _sequence_close(mapped["center_m"], expected_xy)
        )

    fixed_guide = _ring_measurements(assembly_text, "ContinuousGuideRing")
    plug_guide = _ring_measurements(assembly_text, "ContinuousPlugGuide")
    fixed_stop = _ring_measurements(assembly_text, "MetalStop")
    geometry = contract["keying"]["nominal_collision_geometry"]
    stop = contract["metal_stop"]
    shell_dimensions_equal = all(
        (
            _close(fixed_guide["inner_radius_m"], geometry["receptacle_clear_bore_radius_m"]),
            _close(fixed_guide["outer_radius_m"], geometry["receptacle_outer_radius_m"]),
            _close(fixed_guide["z_min_m"], geometry["keyway_local_z_interval_m"][0]),
            _close(fixed_guide["z_max_m"], geometry["keyway_local_z_interval_m"][1]),
            _close(plug_guide["inner_radius_m"], geometry["plug_shell_radial_interval_m"][0]),
            _close(plug_guide["outer_radius_m"], geometry["plug_shell_radial_interval_m"][1]),
            _close(plug_guide["z_min_m"], 0.0),
            _close(plug_guide["z_max_m"], stop["nominal_bottoming_separation_m"]),
        )
    )

    key_angles = [
        _number(_prim_block(assembly_text, "Cube", f"Key_{index}"), "xformOp:rotateZ")
        for index in range(5)
    ]
    keyway_angles = []
    for index in range(5):
        for side in ("LeftWall", "RightWall"):
            keyway_angles.append(
                _number(
                    _prim_block(assembly_text, "Cube", f"Keyway_{index}_{side}"),
                    "xformOp:rotateZ",
                )
            )
    expected_key_angles = [
        float(value)
        for value in contract["keying"]["plug_local_authored_angles_deg"]["values"]
    ]
    expected_keyway_angles = [
        float(angle)
        for angle in contract["keying"]["canonical_angles_deg"]
        for _ in range(2)
    ]
    five_key_directions_equal = (
        _sequence_close(key_angles, expected_key_angles)
        and _sequence_close(keyway_angles, expected_keyway_angles)
        and mapping["keying"]["canonical_angles_deg"]
        == contract["keying"]["canonical_angles_deg"]
        and mapping["keying"]["plug_local_authored_angles_deg"]
        == contract["keying"]["plug_local_authored_angles_deg"]["values"]
    )

    mass_expected = contract["mass_properties"]["bodies"]
    mass_actual = {
        "fixed_receptacle": _mass_measurement(assembly_text, "FixedReceptacle"),
        "loose_plug_body_assembly": _mass_measurement(assembly_text, "BodyAssembly"),
        "coupling_nut": _mass_measurement(assembly_text, "CouplingNut"),
    }
    mass_and_com_equal = all(
        _mass_equal(mass_actual[key], mass_expected[key]) for key in mass_actual
    ) and mapping["mass_properties"] == contract["mass_properties"]

    event_actual = []
    for event in contract["assembly_events"]["ordered"]:
        name = f"Event_{int(event['ordinal']):02d}_{event['name']}"
        block = _prim_block(assembly_text, "Scope", name)
        event_actual.append(
            {
                "ordinal": int(_number(block, "kcg:ordinal")),
                "name": event["name"],
                "nominal_separation_m": _number(block, "kcg:nominalSeparationM"),
            }
        )
    seven_event_positions_equal = all(
        actual["ordinal"] == int(expected["ordinal"])
        and actual["name"] == expected["name"]
        and _close(actual["nominal_separation_m"], expected["nominal_separation_m"])
        for actual, expected in zip(event_actual, contract["assembly_events"]["ordered"])
    ) and mapping["assembly_events"] == contract["assembly_events"]["ordered"]

    thread_block = _prim_block(assembly_text, "Scope", "ThreeStartThread")
    thread_lead_m = _number(thread_block, "kcg:leadMPerRevolution")
    thread_lead_equal = (
        int(_number(thread_block, "kcg:starts")) == int(contract["thread"]["starts"])
        and _close(thread_lead_m, float(contract["thread"]["lead_mm_per_revolution"]) * 1.0e-3)
        and _close(
            mapping["thread"]["lead_mm_per_revolution"],
            contract["thread"]["lead_mm_per_revolution"],
        )
    )
    plug_stop_block = _prim_block(assembly_text, "Cylinder", "MetalStop")
    plug_stop_center = _vector(plug_stop_block, "xformOp:translate")
    metal_stop_equal = all(
        (
            _close(fixed_stop["outer_radius_m"], stop["assembly_control_collision"]["fixed_cap_radius_m"]),
            _close(fixed_stop["z_min_m"], stop["receptacle_stop_surface_depth_from_receptacle_B_m"]),
            _close(
                plug_stop_center[2], stop["plug_stop_surface_depth_from_plug_B_m"]
            ),
            _close(
                _number(plug_stop_block, "radius"),
                stop["assembly_control_collision"]["plug_stop_distribution_radius_m"],
            ),
            mapping["metal_stop"] == stop,
        )
    )

    assembly_paths = _prim_paths(assembly_text)
    expected_paths = _expected_logical_paths(contract, pair_root, labels)
    missing_paths = sorted(expected_paths - assembly_paths)
    mapping_paths = representation["logical_paths"]
    path_mapping_complete = (
        not missing_paths
        and mapping_paths["pair_root"] == pair_root
        and set(mapping_paths) == set(contract["logical_path_interface"]["logical_parts"])
    )
    pair_relations_same_label = _pair_relations_valid(assembly_text, labels, pair_root)
    forbidden_tokens = (
        "hard_socket_entries_61",
        "pin_barriers_61",
        "six_fixed_annular_wedge_petals_per_socket",
        "segmented_axisymmetric_pin_barrier_contact",
    )
    forbidden_hits = [token for token in forbidden_tokens if token in assembly_text]
    cross_label_effect_count = len(representation["cross_label_effects"])
    all_combination_count = int(
        mapping["static_generation_checks"]["all_combination_micro_contact_count"]
    )
    traceable_pair_count = int(representation["pair_effect_count"])

    truth = contract["truth_firewall"]
    truth_firewall_pass = (
        truth["internal_model_state_exposed_to_task_controller"] is False
        and "custom bool kcg:controllerTruthInputAllowed = 0" in assembly_text
        and "custom bool kcg:postPhysicsPoseWriteAllowed = 0" in assembly_text
        and "custom bool kcg:magneticAttractionAllowed = 0" in assembly_text
        and "custom bool kcg:controllerVisible = 0" in assembly_text
        and mapping["truth_firewall"] == truth
    )
    visual_detail_preserved = (
        direct_reference_equal
        and "custom bool kcg:visibleGeometryPreserved = 1" in visual_text
        and "custom int kcg:visiblePinCount = 61" in visual_text
        and "custom int kcg:visibleSocketCount = 61" in visual_text
        and "custom bool kcg:visibleThreadPreserved = 1" in visual_text
        and "custom bool kcg:visibleDetentTeethPreserved = 1" in visual_text
        and "custom bool kcg:visibleSealsPreserved = 1" in visual_text
    )
    assembly_axis_equal = (
        mapping["assembly_axis"] == contract["coordinate_frames"]["assembly_axis"]
        and contract["coordinate_frames"]["assembly_axis"]["receptacle_local_vector"]
        == [0.0, 0.0, 1.0]
        and contract["coordinate_frames"]["assembly_axis"]["local_z_axes_antiparallel_at_mating"]
        is True
    )
    pin_centers_equal = (
        pair_labels == labels
        and len(set(labels)) == 61
        and all(pin_center_checks)
        and all(socket_center_checks)
        and all(mapped_center_checks)
    )

    checks = {
        "input_hashes_unchanged": inputs_unchanged,
        "shared_contract_sha256_equal": contract_link_equal,
        "shared_generator_sha256_equal": generator_link_equal,
        "assembly_axis_equal": assembly_axis_equal,
        "shell_dimensions_equal": shell_dimensions_equal,
        "five_key_directions_equal": five_key_directions_equal,
        "pin_centers_61_equal": pin_centers_equal,
        "mass_and_com_equal": mass_and_com_equal,
        "seven_event_positions_equal": seven_event_positions_equal,
        "thread_lead_equal": thread_lead_equal,
        "metal_stop_equal": metal_stop_equal,
        "path_mapping_complete": path_mapping_complete,
        "pair_relations_same_label": pair_relations_same_label,
        "cross_label_effect_count_zero": cross_label_effect_count == 0,
        "all_combination_micro_contact_count_zero": all_combination_count == 0,
        "traceable_pair_effect_count_61": traceable_pair_count == 61,
        "forbidden_high_detail_contact_families_absent": not forbidden_hits,
        "visual_detail_preserved": visual_detail_preserved,
        "local_reference_direct_and_read_only": direct_reference_equal
        and "custom bool kcg:readOnlyReference = 1" in local_text
        and "custom bool kcg:completeRobotAssemblyMainlineAllowed = 0" in local_text,
        "truth_firewall_pass": truth_firewall_pass,
        "simulation_not_started": build["simulation_started"] is False,
        "formal_p1_not_claimed": build["formal_p1_pass_claimed"] is False,
        "formal_r12_not_generated": build["formal_r12_generated"] is False,
        "hardware_not_authorized": build["hardware_authorized"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": "kcg_task_r12_multilayer_static_consistency_v1",
        "task_id": "TASK-R12-MULTILAYER-004",
        "phase": arguments.phase,
        "validated_at_utc": _utc_now(),
        "status": "PASS" if not failed else "FAIL",
        "classification": "STATIC_CONSISTENCY_CONFIRMED" if not failed else "STATIC_CONSISTENCY_FAILED",
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "failed_checks": failed,
        "checks": checks,
        "input_sha256": {
            CONTRACT_RELATIVE_PATH.as_posix(): _sha256(contract_path),
            BUILD_RESULT_RELATIVE_PATH.as_posix(): _sha256(build_result_path),
            **{
                str(path.relative_to(WORKSPACE_ROOT)): digest
                for path, digest in (
                    (visual_path, actual_hashes[visual_path.name]),
                    (assembly_path, actual_hashes[assembly_path.name]),
                    (local_path, actual_hashes[local_path.name]),
                    (mapping_path, actual_hashes[mapping_path.name]),
                    (expected_reference, _sha256(expected_reference)),
                )
            },
        },
        "measurements": {
            "fixed_continuous_guide": fixed_guide,
            "plug_continuous_guide": plug_guide,
            "fixed_metal_stop": fixed_stop,
            "plug_metal_stop_center_m": plug_stop_center,
            "key_angles_deg": key_angles,
            "keyway_wall_angles_deg": keyway_angles,
            "mass_properties": mass_actual,
            "assembly_events": event_actual,
            "thread_lead_m_per_revolution": thread_lead_m,
        },
        "pair_audit": {
            "contract_pair_count": len(labels),
            "mapping_pair_count": len(pair_labels),
            "traceable_pair_effect_count": traceable_pair_count,
            "same_label_relation_sets_checked": 3 * len(labels),
            "cross_label_effect_count": cross_label_effect_count,
            "all_combination_micro_contact_count": all_combination_count,
            "missing_logical_paths": missing_paths,
            "forbidden_high_detail_family_hits": forbidden_hits,
        },
        "proof_chains": {
            "visual_and_local_structure": "两层直接引用同一摘要冻结高精细资产，未复制或删减其可见结构",
            "assembly_structure": "直接测量控制层USDA并逐项与唯一主合同比较",
            "mass_and_com_across_layers": "外观和局部层引用同一高精细源；控制层质量属性直接测量并与同一主合同相等",
            "controller_isolation": "控制层仅发布内部同编号相对位移作用且全部controllerVisible=false；禁止输入和位姿写入标志保持false",
        },
        "simulation_started": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    _write_result(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
