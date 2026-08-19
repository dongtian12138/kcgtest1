"""Pure-CPU fail-closed contract for the model-first D38999 rebuild.

The contract separates public dimensions from simulation-only proxies and
keeps every keyed-v2 episode blocked until A0 through A5 are complete.  This
module imports neither Isaac Sim nor Pixar USD and computes no file digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

import yaml

from kcg_connector.d38999_keyed_v2_frozen_contract_snapshot import (
    FROZEN_A0_RESOLVED_DECISIONS,
    FROZEN_A0_RESOLVED_SOURCE_MAPPINGS,
    FROZEN_A2_COLLISION_AUTHORING_BLUEPRINT,
    FROZEN_CONVEX_COOKING_REPRESENTATION,
    FROZEN_MODEL_IMMUTABLE_SECTIONS,
    FROZEN_REALIZED_ROBOT_HAND_FIXTURE_BLUEPRINT,
)


SCHEMA_VERSION = "kcg_d38999_keyed_v2_physical_model_contract_v1"
SUCCESSOR_SCHEMA = "kcg_d38999_keyed_physical_v3_v1"
SUCCESSOR_REVISION = "keyed_v3_physical_r11"
SUCCESSOR_ASSET_NAME = "d38999_shell25j_25_61_n_keyed_physical_v3_r11.usda"
SUCCESSOR_ROOT_PRIM = "/World/D38999Shell25JKeyedPhysicalV3"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v2_physical_model_contract_v1.yaml"
)

SOURCE_KINDS = frozenset(
    {
        "PUBLIC_SPEC_VERIFIED",
        "PUBLIC_SPEC_DERIVED",
        "FROZEN_SIMULATION_PROXY",
        "INHERITED_PROXY_TO_VALIDATE",
        "UNMODELED_BLOCKER",
    }
)
REQUIRED_PHASES = ("A0", "A1", "A2", "A3", "A4", "A5")
FINAL_PHASE_STATE = {
    "A0": "FROZEN",
    "A1": "AUDIT_COMPLETE",
    "A2": "ACCEPTED",
    "A3": "ACCEPTED",
    "A4": "ACCEPTED",
    "A5": "FROZEN",
}
REQUIRED_BENCH_IDS = tuple(f"P{index}" for index in range(1, 15))
REQUIRED_SEQUENCE_EVENTS = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)
REQUIRED_SEQUENCE_PRECEDENCE = (
    ("five_key_polarization", "three_start_thread_entry"),
    ("five_key_polarization", "first_pin_socket_spring_touch"),
    ("spring_finger_engagement", "first_pin_socket_spring_touch"),
    ("three_start_thread_entry", "shell_to_shell_metal_bottoming"),
    ("first_pin_socket_spring_touch", "shell_to_shell_metal_bottoming"),
    ("first_pin_socket_spring_touch", "pin_barrier_seal_contact"),
    ("pin_barrier_seal_contact", "shell_to_shell_metal_bottoming"),
    ("seal_compression", "shell_to_shell_metal_bottoming"),
)
NOMINAL_EVENT_B_SEPARATION_MM = {
    "five_key_polarization": 6.50,
    "three_start_thread_entry": 7.43,
    "spring_finger_engagement": 10.80,
    "first_pin_socket_spring_touch": 12.02,
    "pin_barrier_seal_contact": 14.305,
    "seal_compression": 14.615,
    "shell_to_shell_metal_bottoming": 15.05,
}
EXPECTED_CONTACT_POSITIONS_IN = (
    ("A", 0.196, 0.500), ("B", 0.314, 0.435), ("C", 0.413, 0.343),
    ("D", 0.485, 0.230), ("E", 0.527, 0.101), ("F", 0.536, -0.030),
    ("G", 0.511, -0.164), ("H", 0.454, -0.287), ("J", 0.368, -0.391),
    ("K", 0.259, -0.470), ("L", 0.134, -0.519), ("M", 0.000, -0.537),
    ("N", -0.134, -0.519), ("P", -0.259, -0.470), ("R", -0.368, -0.391),
    ("S", -0.454, -0.287), ("T", -0.511, -0.164), ("U", -0.536, -0.030),
    ("V", -0.527, 0.101), ("W", -0.485, 0.230), ("X", -0.413, 0.343),
    ("Y", -0.314, 0.435), ("Z", -0.196, 0.500), ("a", -0.068, 0.454),
    ("b", 0.068, 0.454), ("c", 0.173, 0.363), ("d", 0.285, 0.283),
    ("e", 0.362, 0.175), ("f", 0.399, 0.046), ("g", 0.392, -0.088),
    ("h", 0.341, -0.213), ("i", 0.251, -0.314), ("j", 0.133, -0.379),
    ("k", 0.000, -0.402), ("m", -0.133, -0.379), ("n", -0.251, -0.314),
    ("p", -0.341, -0.213), ("q", -0.392, -0.088), ("r", -0.399, 0.046),
    ("s", -0.362, 0.175), ("t", -0.285, 0.283), ("u", -0.173, 0.363),
    ("v", 0.000, 0.338), ("w", 0.147, 0.223), ("x", 0.237, 0.122),
    ("y", 0.267, -0.010), ("z", 0.228, -0.139), ("AA", 0.131, -0.233),
    ("BB", 0.000, -0.267), ("CC", -0.131, -0.233), ("DD", -0.228, -0.139),
    ("EE", -0.267, -0.010), ("FF", -0.237, 0.122), ("GG", -0.147, 0.223),
    ("HH", 0.000, 0.200), ("JJ", 0.105, 0.094), ("KK", 0.135, -0.041),
    ("LL", 0.000, -0.132), ("MM", -0.135, -0.041), ("NN", -0.105, 0.094),
    ("PP", 0.000, 0.000),
)
REQUIRED_MATERIAL_ROLES = frozenset(
    {
        "fingertip_pad",
        "finger_structure",
        "robot_structure",
        "coupling_nut_outer_grip",
        "coupling_bearing_and_shoulder",
        "plug_shell_and_keys",
        "pin_and_socket",
        "coupling_thread",
        "spring_finger",
        "anti_decoupling_detent",
        "interfacial_pin_barrier",
        "peripheral_seal",
        "table",
        "fixture_and_receptacle",
    }
)
EXPECTED_MATERIAL_ROLE_VALUES = {
    "fingertip_pad": {"static_friction": 1.40, "dynamic_friction": 1.40, "restitution": 0.0},
    "finger_structure": {"static_friction": 0.35, "dynamic_friction": 0.25, "restitution": 0.0},
    "robot_structure": {"static_friction": 0.35, "dynamic_friction": 0.25, "restitution": 0.0},
    "coupling_nut_outer_grip": {"static_friction": 0.55, "dynamic_friction": 0.45, "restitution": 0.0},
    "coupling_bearing_and_shoulder": {"static_friction": 0.10, "dynamic_friction": 0.08, "restitution": 0.0},
    "plug_shell_and_keys": {"static_friction": 0.35, "dynamic_friction": 0.25, "restitution": 0.0},
    "pin_and_socket": {"static_friction": 0.30, "dynamic_friction": 0.20, "restitution": 0.0},
    "coupling_thread": {"static_friction": 0.20, "dynamic_friction": 0.15, "restitution": 0.0},
    "spring_finger": {"static_friction": 0.25, "dynamic_friction": 0.20, "restitution": 0.0},
    "anti_decoupling_detent": {"static_friction": 0.0, "dynamic_friction": 0.0, "restitution": 0.0},
    "interfacial_pin_barrier": {"static_friction": 0.80, "dynamic_friction": 0.65, "restitution": 0.0},
    "peripheral_seal": {"static_friction": 0.80, "dynamic_friction": 0.65, "restitution": 0.0},
    "table": {"static_friction": 0.90, "dynamic_friction": 0.75, "restitution": 0.0},
    "fixture_and_receptacle": {"static_friction": 0.35, "dynamic_friction": 0.25, "restitution": 0.0},
}
REQUIRED_HARD_RESPONSE_ROLES = frozenset(
    {
        "hard_rigid", "hard_thread", "hard_pin", "hard_socket_entry",
        "hard_insert_face", "hard_receptacle_backing_face",
        "hard_receptacle_keyway_shell", "hard_receptacle_thread_carrier",
        "hard_plug_mating_shell", "hard_polarizing_key", "hard_coupling_nut",
        "hard_plug_rear_body", "hard_receptacle_bore", "hard_seal_target",
        "hard_detent_cam", "hard_metal_bottoming", "hard_nut_body_shoulder",
    }
)
REQUIRED_COMPLIANT_RESPONSE_VALUES = {
    "compliant_socket_petal": (4000.0, 0.10),
    "compliant_spring_finger": (12000.0, 1.0),
    "compliant_pin_barrier": (10.416666666666666, 0.0008333333333333334),
    "compliant_peripheral_seal": (800.0, 0.5),
    "compliant_detent_follower": (110000.0, 2.0),
}
REQUIRED_FORCE_PROXY_ROLES = frozenset(
    {
        "socket_contact_sleeves_61",
        "shell_spring_fingers",
        "interfacial_pin_barriers_61",
        "peripheral_seal",
        "anti_decoupling_detent",
        "coupling_thread",
    }
)
REQUIRED_COMPONENT_IDS = frozenset(
    {
        "receptacle_mating_shell",
        "receptacle_flange_and_mount",
        "plug_mating_shell",
        "five_keys",
        "five_keyways",
        "insert_faces_with_61_clear_holes",
        "pins_61",
        "socket_entries_61",
        "compliant_contact_sleeves_61",
        "coupling_nut_annular_grip",
        "nut_to_body_retaining_shoulder",
        "nut_body_cylindrical_bearing",
        "three_start_thread",
        "spring_fingers",
        "pin_barriers_61",
        "peripheral_seal",
        "shell_to_shell_bottoming",
        "anti_decoupling",
        "red_band",
        "robot_and_hand_rigid_bodies",
        "robot_self_collision",
        "fixture_load_path",
    }
)
REQUIRED_SELF_COLLISION_EXCLUSIONS = frozenset(
    {
        tuple(sorted(pair))
        for pair in (
            ("iiwa_link_0", "iiwa_link_1"),
            ("iiwa_link_1", "iiwa_link_2"),
            ("iiwa_link_2", "iiwa_link_3"),
            ("iiwa_link_3", "iiwa_link_4"),
            ("iiwa_link_4", "iiwa_link_5"),
            ("iiwa_link_5", "iiwa_link_6"),
            ("iiwa_link_6", "iiwa_link_7"),
            ("iiwa_link_7", "handbase_link"),
            ("handbase_link", "f1Link1"),
            ("f1Link1", "f1Link2"),
            ("f1Link2", "f1Link3"),
            ("handbase_link", "f2Link1"),
            ("f2Link1", "f2Link2"),
            ("handbase_link", "f3Link1"),
            ("f3Link1", "f3Link2"),
            ("f3Link2", "f3Link3"),
        )
    }
)
FORBIDDEN_METADATA_KEY_PARTS = ("sha256", "checksum", "digest", "hash")
DOWNSTREAM_AUTHORIZATION_FIELDS = (
    "grasp_allowed",
    "camera_dataset_allowed",
    "visual_control_allowed",
    "insertion_allowed",
    "twist_allowed",
    "randomization_allowed",
    "training_allowed",
    "rl_allowed",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_exact_fields(
    parent: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for field, required in expected.items():
        if parent.get(field) != required:
            raise ValueError(f"{label}.{field} differs from the frozen blueprint")


def _require_exact_numeric_tree(
    actual: Any,
    expected: Any,
    label: str,
    *,
    abs_tol: float = 1.0e-15,
) -> None:
    """Compare a frozen nested structure without accepting omitted or extra leaves."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} mapping inventory differs from the frozen blueprint")
        for key, expected_child in expected.items():
            _require_exact_numeric_tree(
                actual[key], expected_child, f"{label}.{key}", abs_tol=abs_tol
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list inventory differs from the frozen blueprint")
        for index, expected_child in enumerate(expected):
            _require_exact_numeric_tree(
                actual[index], expected_child, f"{label}[{index}]", abs_tol=abs_tol
            )
        return
    if isinstance(expected, Real) and not isinstance(expected, bool):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, Real)
            or not math.isclose(float(actual), float(expected), abs_tol=abs_tol)
        ):
            raise ValueError(f"{label} differs from the frozen numeric blueprint")
        return
    if actual != expected:
        raise ValueError(f"{label} differs from the frozen blueprint")


def _walk_mapping_keys(value: Any, prefix: str = "document") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{prefix} contains a non-text key")
            path = f"{prefix}.{key}"
            yield path
            yield from _walk_mapping_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping_keys(child, f"{prefix}[{index}]")


def _reject_fingerprint_metadata(document: Mapping[str, Any]) -> None:
    for path in _walk_mapping_keys(document):
        leaf = path.rsplit(".", 1)[-1].lower().replace("-", "_")
        if any(part in leaf for part in FORBIDDEN_METADATA_KEY_PARTS):
            raise ValueError(
                "physical-model evidence must use semantic identity and "
                f"resolved readback, not fingerprint metadata: {path}"
            )


def _validate_sources(document: Mapping[str, Any]) -> None:
    classes = _mapping(document.get("source_classes"), "source_classes")
    declared = classes.get("allowed")
    if not isinstance(declared, list) or frozenset(declared) != SOURCE_KINDS:
        raise ValueError("source_classes.allowed differs from the frozen taxonomy")
    if classes.get("hardware_truth_requires_measurement") is not True:
        raise ValueError("hardware truth must require measurement")
    sources = _mapping(document.get("public_sources"), "public_sources")
    required_sources = {
        "general_interface",
        "insert_pattern",
        "receptacle_detail",
        "plug_detail",
        "robot_active_description",
    }
    if set(sources) != required_sources:
        raise ValueError("public_sources are incomplete")
    for name, value in sources.items():
        source = _mapping(value, f"public_sources.{name}")
        if source.get("source_kind") not in SOURCE_KINDS:
            raise ValueError(f"public_sources.{name}.source_kind is invalid")


def _validate_identity(document: Mapping[str, Any]) -> None:
    identity = _mapping(document.get("identity"), "identity")
    expected = {
        "task_lineage": "keyed_v2",
        "predecessor_revision": "keyed_v3_physical_r10",
        "predecessor_role":
        "rejected_runtime_detent_settle_margin_baseline_never_modify",
        "successor_schema": SUCCESSOR_SCHEMA,
        "successor_revision": SUCCESSOR_REVISION,
        "root_prim": SUCCESSOR_ROOT_PRIM,
        "recommended_asset_name": SUCCESSOR_ASSET_NAME,
    }
    for field, required in expected.items():
        if identity.get(field) != required:
            raise ValueError(f"identity.{field} must be {required}")
    if identity.get("overwrite_existing") is not False:
        raise ValueError("the successor asset must refuse overwrite")
    if identity.get("immutable_after_a5") is not True:
        raise ValueError("the successor asset must be immutable after A5")
    if "r6" in str(identity.get("recommended_asset_name", "")).lower():
        raise ValueError("the successor asset cannot reuse the r6 identity")


def _validate_convex_cooking_representation(document: Mapping[str, Any]) -> None:
    representation = _mapping(
        document.get("convex_cooking_representation"),
        "convex_cooking_representation",
    )
    if representation != FROZEN_CONVEX_COOKING_REPRESENTATION:
        raise ValueError(
            "convex cooking representation differs from its independent frozen literal"
        )
    multiplier = _finite(
        representation.get("authored_mesh_point_multiplier_from_blueprint"),
        "convex cooking point multiplier",
    )
    scale = representation.get("mesh_uniform_scale_xyz")
    if scale != [0.001, 0.001, 0.001] or not math.isclose(
        multiplier * float(scale[0]), 1.0, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("convex cooking representation changes world geometry")
    local_minimum = _finite(
        representation.get(
            "physxConvexHullCollision:minThickness_local_units"
        ),
        "convex cooking local minimum thickness",
    )
    effective = _finite(
        representation.get("effective_world_min_thickness_m"),
        "convex cooking effective world minimum thickness",
    )
    if not math.isclose(
        local_minimum * float(scale[0]), effective, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("convex cooking minimum-thickness conversion changed")


def _validate_scope_and_evidence(document: Mapping[str, Any]) -> None:
    scope = _mapping(document.get("scope"), "scope")
    if scope.get("target") != "deterministic_public_spec_simulation":
        raise ValueError("physical-model scope must remain deterministic public-spec simulation")
    false_claims = (
        "manufacturer_cad_claimed",
        "manufactured_unit_fidelity_claimed",
        "hardware_force_curve_claimed",
        "qualification_claimed",
    )
    if any(scope.get(field) is not False for field in false_claims):
        raise ValueError("simulation contract was promoted to an unsupported hardware claim")
    if scope.get("future_vendor_cad_or_hardware_recalibration_excluded") is not True:
        raise ValueError("future vendor or measured data must remain a new scope")
    evidence = _mapping(document.get("evidence_policy"), "evidence_policy")
    if evidence.get("cryptographic_fingerprints_allowed") is not False:
        raise ValueError("cryptographic fingerprints are forbidden by this contract")
    if evidence.get("reports_must_be_fail_closed") is not True:
        raise ValueError("physical-model evidence must be fail-closed")


def _validate_a0(document: Mapping[str, Any]) -> None:
    section = _mapping(document.get("a0_source_freeze"), "a0_source_freeze")
    unresolved = section.get("unresolved_source_mappings")
    if not isinstance(unresolved, list):
        raise ValueError("a0_source_freeze.unresolved_source_mappings must be a list")
    unresolved_blockers = []
    seen = set()
    for index, raw in enumerate(unresolved):
        item = _mapping(raw, f"unresolved_source_mappings[{index}]")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("A0 unresolved mapping ids must be unique text")
        seen.add(item_id)
        if item.get("blocking") is True:
            unresolved_blockers.append(item_id)
    frozen = section.get("status") == "FROZEN"
    expected_document_status = (
        "A0_FROZEN_A2_AUTHORIZED" if frozen else "A0_SOURCE_FREEZE_IN_PROGRESS"
    )
    if document.get("status") != expected_document_status:
        raise ValueError("document status differs from the A0 freeze state")
    if frozen and unresolved_blockers:
        raise ValueError("A0 cannot be frozen while source blockers remain")
    if bool(section.get("a2_asset_authoring_allowed")) != frozen:
        raise ValueError("A2 authoring permission must exactly follow A0 freeze")
    decisions = section.get("resolved_decisions")
    if not isinstance(decisions, list) or len(decisions) < 8:
        raise ValueError("A0 resolved decisions are incomplete")
    decision_names = {
        _mapping(item, "resolved decision").get("name") for item in decisions
    }
    required = {
        "simulation_scope",
        "connector_asset_identity",
        "active_robot_description",
        "camera_model",
        "wrist_force_model",
        "finger_sensor_model",
        "fixture_load_path",
        "coupling_thread_architecture",
        "electrical_contact_architecture",
        "self_collision_policy",
        "solver_authoring_api",
        "component_root_and_bottoming_frames",
        "nut_body_bearing_and_load_path",
        "material_role_partition",
    }
    if not required.issubset(decision_names):
        raise ValueError("A0 is missing a required full-chain modeling decision")
    if frozen:
        if not {
            "figure3_axial_mapping",
            "force_transmitting_proxy_parameters",
        }.issubset(decision_names):
            raise ValueError("A0 freeze lacks axial or force-proxy decisions")
        resolved = section.get("resolved_source_mappings")
        if not isinstance(resolved, list):
            raise ValueError("frozen A0 requires resolved_source_mappings")
        resolved_by_id = {
            _mapping(item, "resolved source mapping").get("id"): item
            for item in resolved
        }
        if set(resolved_by_id) != {
            "AXIAL_FIGURE3_DATUM_MAP",
            "CONNECTOR_PROXY_FORCE_PARAMETERS",
        }:
            raise ValueError("A0 resolved source mapping inventory changed")
        if any(
            not isinstance(item.get("status"), str)
            or not item["status"].startswith("RESOLVED_")
            or not isinstance(item.get("boundary"), str)
            for item in resolved_by_id.values()
        ):
            raise ValueError("A0 resolved mappings lack status or scope boundary")


def _validate_coordinate_contract(document: Mapping[str, Any]) -> None:
    coordinates = _mapping(document.get("coordinate_contract"), "coordinate_contract")
    if (
        coordinates.get("stage_units") != "meters"
        or coordinates.get("stage_mass_units") != "kilograms"
        or coordinates.get("pair_root") != SUCCESSOR_ROOT_PRIM
        or coordinates.get("fixed_endpoint_tabletop_orientation_deg_xyz")
        != [180.0, 0.0, 0.0]
        or coordinates.get("loose_endpoint_tabletop_orientation_deg_xyz")
        != [180.0, 0.0, 0.0]
    ):
        raise ValueError("stage units, pair root, or tabletop orientation changed")
    roots = _mapping(coordinates.get("component_roots"), "component_roots")
    expected_roots = {
        "fixed_receptacle": (
            "/FixedReceptacle",
            "receptacle_datum_B_center",
            "from_datum_B_toward_receptacle_interior",
        ),
        "loose_plug": (
            "/LoosePlug",
            "plug_datum_B_center",
            "from_datum_B_toward_plug_rear",
        ),
    }
    if set(roots) != set(expected_roots):
        raise ValueError("component-root inventory changed")
    for name, (suffix, origin, plus_z) in expected_roots.items():
        root = _mapping(roots[name], f"component_roots.{name}")
        if (
            root.get("prim_suffix") != suffix
            or root.get("origin") != origin
            or root.get("local_plus_z") != plus_z
            or root.get("datum_B_translation_m") != [0.0, 0.0, 0.0]
        ):
            raise ValueError(f"{name} datum-B root transform changed")
    mating = _mapping(coordinates.get("mating_transform"), "mating_transform")
    if (
        mating.get("local_z_axes_antiparallel") is not True
        or mating.get("local_x_axes_coincident_at_correct_N_key") is not True
        or mating.get("plug_translation_direction") != "opposite_plug_local_plus_z"
        or _finite(mating.get("full_mate_datum_B_separation_mm"), "full mate B separation")
        != 15.05
        or mating.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("mating transform or full-mate datum-B separation changed")


def _validate_geometry(document: Mapping[str, Any]) -> None:
    geometry = _mapping(document.get("public_geometry"), "public_geometry")
    receptacle = _mapping(
        geometry.get("receptacle_shell_25_class_k"), "receptacle geometry"
    )
    plug = _mapping(geometry.get("plug_shell_25_class_k"), "plug geometry")
    keying = _mapping(geometry.get("keying_n"), "N keying")
    contacts = _mapping(geometry.get("contact_pattern_25_61"), "25-61 contacts")
    thread = _mapping(geometry.get("thread_shell_25"), "shell-25 thread")
    expected_ranges = (
        (receptacle, "mating_shell_f_diameter_mm", 39.02, 39.42),
        (receptacle, "bore_h_diameter_mm", 35.84, 35.99),
        (receptacle, "keyway_j_diameter_mm", 37.92, 38.20),
        (plug, "inner_u_diameter_mm", 33.99, 34.34),
        (plug, "between_keys_w_diameter_mm", 35.61, 35.77),
        (plug, "over_keys_v_diameter_mm", 37.57, 37.85),
    )
    for parent, field, minimum, maximum in expected_ranges:
        value = _mapping(parent.get(field), field)
        if (_finite(value.get("min"), f"{field}.min"), _finite(value.get("max"), f"{field}.max")) != (minimum, maximum):
            raise ValueError(f"{field} differs from the public shell-25 range")
    if keying.get("angles_deg") != [0.0, 80.0, 142.0, 196.0, 293.0]:
        raise ValueError("N polarization angles changed")
    if keying.get("polarization_precedes_thread_and_contact") is not True:
        raise ValueError("polarization must precede thread and contact")
    if contacts.get("exact_contact_count") != 61:
        raise ValueError("25-61 must contain exactly 61 physical contacts")
    if contacts.get("same_label_pairing_required") is not True:
        raise ValueError("61 contacts must preserve same-label pairing")
    detail = _mapping(
        contacts.get("series_III_size20_interface_detail"),
        "Series-III size-20 interface detail",
    )
    if (
        detail.get("source_kind") != "PUBLIC_SPEC_VERIFIED"
        or detail.get("source_locator")
        != "MIL-DTL-38999N_Figure_14_series_III_size20"
        or detail.get("public_dimensions_do_not_define_force_curve") is not True
    ):
        raise ValueError("Figure-14 size-20 source boundary changed")
    for field, minimum, nominal, maximum in (
        ("socket_entry_F_diameter_mm", 1.24, 1.28, 1.32),
        ("socket_entry_G_diameter_mm", 2.16, 2.21, 2.26),
        ("socket_entry_front_lip_axial_mm", 0.00, 0.065, 0.13),
        ("socket_entry_chamfer_angle_deg", 43.0, 45.0, 47.0),
        ("pin_barrier_E_diameter_mm", 2.31, 2.375, 2.44),
        ("pin_barrier_C_diameter_mm", 1.83, 1.87, 1.91),
        ("pin_barrier_total_axial_mm", 1.19, 1.32, 1.45),
        ("pin_barrier_tip_straight_axial_mm", 0.51, 0.575, 0.64),
    ):
        values = _mapping(detail.get(field), f"Figure-14 {field}")
        observed = tuple(
            _finite(values.get(key), f"Figure-14 {field}.{key}")
            for key in ("min", "nominal", "max")
        )
        if observed != (minimum, nominal, maximum):
            raise ValueError(f"Figure-14 {field} changed")
    if (
        _finite(
            detail.get("pin_barrier_D_diameter_max_mm"),
            "Figure-14 pin-barrier D maximum",
        )
        != 0.97
        or _finite(
            detail.get("pin_barrier_D_diameter_r7_proxy_mm"),
            "r7 pin-barrier D proxy",
        )
        != 0.96
    ):
        raise ValueError("Figure-14 pin-barrier D public/proxy boundary changed")
    _validate_size20_interface_blueprint(detail)
    if thread.get("starts") != 3 or thread.get("form") != "modified_60_degree_stub":
        raise ValueError("Series III thread must remain three-start modified stub")
    if not math.isclose(_positive(thread.get("pitch_mm"), "thread pitch"), 2.54):
        raise ValueError("thread pitch must be 2.54 mm")
    if not math.isclose(_positive(thread.get("lead_mm_per_revolution"), "thread lead"), 7.62):
        raise ValueError("thread lead must be 7.62 mm/revolution")
    if (
        _positive(thread.get("lead_in_per_revolution"), "thread lead in inches")
        != 0.3
        or thread.get("lead_in_per_revolution_units") != "inch_per_revolution"
        or "lead_in" in thread
    ):
        raise ValueError("0.3 inch/revolution lead semantics changed")
    sequence = _mapping(geometry.get("mating_sequence"), "mating_sequence")
    if sequence.get("partial_order_only") is not True:
        raise ValueError("public mating sequence must remain a partial order")
    if tuple(sequence.get("events", ())) != REQUIRED_SEQUENCE_EVENTS:
        raise ValueError("mating event inventory changed")
    precedence = tuple(
        tuple(edge) for edge in sequence.get("required_precedence_edges", ())
    )
    if precedence != REQUIRED_SEQUENCE_PRECEDENCE:
        raise ValueError("public mating precedence constraints changed")
    if tuple(sequence.get("r7_nominal_expected_order", ())) != REQUIRED_SEQUENCE_EVENTS:
        raise ValueError("r7 nominal simulation sequence changed")
    if sequence.get("r7_nominal_order_source_kind") != "FROZEN_SIMULATION_PROXY":
        raise ValueError("r7 nominal sequence was promoted to public truth")
    event_positions = _mapping(
        sequence.get("r7_nominal_event_datum_B_separation_mm"),
        "r7 nominal event positions",
    )
    if event_positions.get("source_kind") != "FROZEN_SIMULATION_PROXY":
        raise ValueError("r7 nominal event positions lost proxy classification")
    observed_positions = {
        event: _finite(event_positions.get(event), f"event position {event}")
        for event in REQUIRED_SEQUENCE_EVENTS
    }
    if observed_positions != NOMINAL_EVENT_B_SEPARATION_MM:
        raise ValueError("r7 nominal event positions changed")
    if list(observed_positions.values()) != sorted(observed_positions.values()):
        raise ValueError("r7 nominal event positions are not monotonic")
    if (
        observed_positions["first_pin_socket_spring_touch"]
        - observed_positions["spring_finger_engagement"]
        < 1.02
    ):
        raise ValueError("r7 spring fingers do not lead contact by 1.02 mm")
    if _positive(
        sequence.get("spring_finger_lead_before_electrical_contact_min_mm"),
        "spring finger lead",
    ) < 1.02:
        raise ValueError("spring fingers must lead electrical contact by at least 1.02 mm")
    _validate_axial_interface(geometry)


def _validate_size20_interface_blueprint(detail: Mapping[str, Any]) -> None:
    blueprint = _mapping(
        detail.get("r7_collision_blueprint"), "size-20 collision blueprint"
    )
    socket = _mapping(blueprint.get("socket_entry"), "socket-entry blueprint")
    expected_socket_fields = {
        "count": 61,
        "owner_suffix": "/LoosePlug/BodyAssembly",
        "entry_path_template": "/LoosePlug/BodyAssembly/Contacts/Socket_{label}/HardEntry",
        "convex_piece_path_template": "/LoosePlug/BodyAssembly/Contacts/Socket_{label}/HardEntry/Band_{profile_band_index:02d}/Wedge_{angular_segment_index:02d}",
        "shape": "axisymmetric_hard_annulus_with_front_lip_and_internal_conical_chamfer",
        "convex_wedge_segments_per_entry": 24,
        "outer_carrier_radius_mm": 1.25,
        "explicit_convex_piece_count_per_entry": 72,
        "capped_solid_cylinder_allowed": False,
        "hard_face_local_depth_from_plug_B_mm": 0.14,
        "hard_face_depth_source_kind": "FROZEN_SIMULATION_PROXY",
        "material_role": "pin_and_socket",
        "response_role": "hard_socket_entry",
    }
    if any(socket.get(key) != value for key, value in expected_socket_fields.items()):
        raise ValueError("61 hard socket-entry collider blueprint changed")
    if socket.get("axial_profile_bands") != [
        {"profile_band_index": 0, "name": "front_lip", "depth_start_mm": 0.14,
         "depth_end_mm": 0.205, "inner_radius_start_mm": 1.105,
         "inner_radius_end_mm": 1.105},
        {"profile_band_index": 1, "name": "conical_chamfer", "depth_start_mm": 0.205,
         "depth_end_mm": 0.670, "inner_radius_start_mm": 1.105,
         "inner_radius_end_mm": 0.640},
        {"profile_band_index": 2, "name": "throat_bore", "depth_start_mm": 0.670,
         "depth_end_mm": 2.00, "inner_radius_start_mm": 0.640,
         "inner_radius_end_mm": 0.640},
    ]:
        raise ValueError("socket-entry convex axial profile bands changed")
    chamfer_band = socket["axial_profile_bands"][1]
    if not math.isclose(
        chamfer_band["depth_end_mm"] - chamfer_band["depth_start_mm"],
        chamfer_band["inner_radius_start_mm"] - chamfer_band["inner_radius_end_mm"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("nominal socket-entry chamfer is not 45 degrees")
    barrier = _mapping(blueprint.get("pin_barrier"), "pin-barrier blueprint")
    expected_barrier_fields = {
        "count": 61,
        "owner_suffix": "/FixedReceptacle",
        "barrier_path_template": "/FixedReceptacle/Contacts/Barrier_{label}",
        "convex_piece_path_template": "/FixedReceptacle/Contacts/Barrier_{label}/Band_{profile_band_index:02d}/Wedge_{angular_segment_index:02d}",
        "shape": "axisymmetric_annular_tip_cylinder_plus_conical_flare",
        "convex_wedge_segments_per_barrier": 24,
        "explicit_convex_piece_count_per_barrier": 48,
        "capped_solid_cylinder_allowed": False,
        "tip_front_depth_from_receptacle_B_mm": 13.93,
        "base_depth_from_receptacle_B_mm": 15.25,
        "inner_D_diameter_mm": 0.96,
        "tip_outer_C_diameter_mm": 1.87,
        "base_outer_E_diameter_mm": 2.375,
        "tip_straight_axial_mm": 0.575,
        "flare_axial_mm": 0.745,
        "material_role": "interfacial_pin_barrier",
        "response_role": "compliant_pin_barrier",
        "response_parameter_scope": "per_angular_wedge_shared_by_its_nonoverlapping_profile_bands",
    }
    if any(barrier.get(key) != value for key, value in expected_barrier_fields.items()):
        raise ValueError("61 compliant pin-barrier collider blueprint changed")
    if barrier.get("axial_profile_bands") != [
        {"profile_band_index": 0, "name": "tip_straight", "depth_start_mm": 13.93,
         "depth_end_mm": 14.505, "inner_radius_start_mm": 0.480,
         "inner_radius_end_mm": 0.480, "outer_radius_start_mm": 0.935,
         "outer_radius_end_mm": 0.935},
        {"profile_band_index": 1, "name": "conical_flare", "depth_start_mm": 14.505,
         "depth_end_mm": 15.25, "inner_radius_start_mm": 0.480,
         "inner_radius_end_mm": 0.480, "outer_radius_start_mm": 0.935,
         "outer_radius_end_mm": 1.1875},
    ]:
        raise ValueError("pin-barrier convex axial profile bands changed")
    shared_recipe = _mapping(
        blueprint.get("shared_wedge_recipe"), "size-20 shared wedge recipe"
    )
    if shared_recipe != {
        "corner_template_id": "axial_band_direct_endpoint_profile_v1",
        "angular_segment_count": 24,
        "angular_step_deg": 15.0,
        "phase_origin_deg": 0.0,
        "segment_zero_interval_deg": [-7.5, 7.5],
        "radial_arc_approximation": "chord_between_exact_endpoint_rays",
        "vertex_order": "deterministic_lexicographic_profile_then_theta",
        "face_topology": "closed_convex_prism_between_two_profile_cross_sections",
        "adjacent_segments_share_exact_seam_vertices": True,
        "adjacent_profile_bands_share_exact_seam_vertices": True,
        "seam_gap_m": 0.0,
        "seam_overlap_m": 0.0,
        "automatic_convex_decomposition_allowed": False,
        "physics_approximation": "convexHull",
        "collision_offset_class": "fine_connector",
        "local_xy_center_source": "contact_pattern_25_61_source_config",
    }:
        raise ValueError("size-20 explicit convex wedge recipe changed")
    if not math.isclose(
        barrier["tip_straight_axial_mm"] + barrier["flare_axial_mm"],
        barrier["base_depth_from_receptacle_B_mm"]
        - barrier["tip_front_depth_from_receptacle_B_mm"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("pin-barrier tip and flare do not close the axial profile")
    if (
        blueprint.get("first_contact_feature_pair")
        != "pin_barrier_C_outer_cylinder_to_socket_entry_chamfer"
        or blueprint.get("nominal_first_contact_formula")
        != "barrier_tip_depth+plug_face_depth+lip+(G-C)/2/tan(chamfer_angle)"
        or _finite(
            blueprint.get("nominal_first_contact_datum_B_separation_mm"),
            "pin-barrier nominal collider first contact",
        )
        != 14.305
        or blueprint.get("first_contact_must_be_derived_from_collision_geometry_bounds")
        is not True
        or blueprint.get("variant_selection_source_kind")
        != "FROZEN_SIMULATION_PROXY"
        or blueprint.get("public_tolerance_correlation_claimed") is not False
        or blueprint.get("unreached_profile_cap_is_not_force_or_energy_input")
        is not True
    ):
        raise ValueError("pin-barrier/socket-entry first-contact contract changed")
    variants = _mapping(
        blueprint.get("deterministic_geometry_variants"),
        "size-20 deterministic geometry variants",
    )
    expected_inputs = {
        "minimum_interference": {
            "N_mm": 15.04,
            "barrier_total_axial_mm": 1.19,
            "plug_B_to_socket_face_mm": 0.33,
            "lip_mm": 0.00,
            "F_mm": 1.32,
            "G_mm": 2.16,
            "C_mm": 1.83,
            "E_mm": 2.31,
            "chamfer_angle_deg": 47.0,
            "bottoming_stop_separation_mm": 15.01,
        },
        "nominal": {
            "N_mm": 15.25,
            "barrier_total_axial_mm": 1.32,
            "plug_B_to_socket_face_mm": 0.14,
            "lip_mm": 0.065,
            "F_mm": 1.28,
            "G_mm": 2.21,
            "C_mm": 1.87,
            "E_mm": 2.375,
            "chamfer_angle_deg": 45.0,
            "bottoming_stop_separation_mm": 15.05,
        },
        "maximum_interference": {
            "N_mm": 15.46,
            "barrier_total_axial_mm": 1.45,
            "plug_B_to_socket_face_mm": -0.05,
            "lip_mm": 0.13,
            "F_mm": 1.24,
            "G_mm": 2.26,
            "C_mm": 1.91,
            "E_mm": 2.44,
            "chamfer_angle_deg": 43.0,
            "bottoming_stop_separation_mm": 15.09,
        },
    }
    variant_controls = {
        "application_rule":
        "replace_only_fields_explicitly_listed_below_all_other_r7_geometry_is_nominal",
        "output_identity_suffixes_exactly": [
            "minimum_interference", "nominal", "maximum_interference"
        ],
        "invariant_nominal_r7_fields": {
            "barrier_tip_straight_axial_mm": 0.575,
            "barrier_inner_D_diameter_mm": 0.96,
            "socket_throat_depth_from_plug_B_mm": 2.00,
            "socket_petal_depth_breakpoints_mm": [2.00, 2.20, 2.80],
            "receptacle_pin_tip_M_mm": 10.02,
            "hard_insert_face_outer_radius_mm": 15.10,
        },
        "propagation_contract": {
            "authored_collider_values_use_SI_units": True,
            "authored_values_below_override_the_nominal_A2_blueprint": True,
            "all_unlisted_A2_geometry_fields_remain_nominal_r7": True,
            "N_drives": [
                "pin_end_center_and_height", "barrier_base",
                "fixed_backing_face_interval",
            ],
            "barrier_total_axial_drives": ["barrier_tip_front"],
            "invariant_tip_straight_axial_drives": [
                "barrier_tip_to_flare_break"
            ],
            "plug_B_to_socket_face_drives": [
                "socket_entry_front", "hard_insert_face_interval"
            ],
            "lip_G_F_and_chamfer_drive": ["socket_entry_profile_bands"],
            "G_drives": [
                "socket_entry_mouth_radius", "hard_insert_face_hole_clearance"
            ],
            "C_and_E_drive": ["barrier_tip_and_base_outer_radii"],
            "E_drives": ["fixed_backing_face_hole_clearance"],
            "bottoming_stop_separation_drives": ["plug_metal_stop_interval"],
            "polygon_hole_vertex_radius_formula":
            "required_clear_radius/cos(7.5deg)",
            "zero_axial_length_socket_entry_band_policy":
            "omit_band_and_its_24_wedges",
            "omitted_band_indices_by_variant": {
                "minimum_interference": [0], "nominal": [],
                "maximum_interference": [],
            },
            "invariant_socket_petal_geometry": True,
            "invariant_contact_XY_layout": True,
            "output_root_prim": SUCCESSOR_ROOT_PRIM,
            "output_variant_metadata_attribute": "kcg:geometryVariant",
            "output_asset_paths": {
                "minimum_interference":
                "artifacts/kcg_connector/isaac/keyed_v3_physical_r11/"
                "d38999_shell25j_25_61_n_keyed_physical_v3_r11_minimum_interference.usda",
                "nominal":
                "artifacts/kcg_connector/isaac/keyed_v3_physical_r11/"
                "d38999_shell25j_25_61_n_keyed_physical_v3_r11.usda",
                "maximum_interference":
                "artifacts/kcg_connector/isaac/keyed_v3_physical_r11/"
                "d38999_shell25j_25_61_n_keyed_physical_v3_r11_maximum_interference.usda",
            },
        },
    }
    for field, expected in variant_controls.items():
        if variants.get(field) != expected:
            raise ValueError(f"size-20 variant control {field} changed")
    if set(variants) != set(expected_inputs) | set(variant_controls):
        raise ValueError("size-20 deterministic variant inventory changed")
    for name, frozen in expected_inputs.items():
        variant = _mapping(variants[name], f"size-20 variant {name}")
        for field, expected in frozen.items():
            if not math.isclose(
                _finite(variant.get(field), f"{name}.{field}"),
                expected,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"size-20 variant {name}.{field} changed")
        barrier_front = frozen["N_mm"] - frozen["barrier_total_axial_mm"]
        chamfer_depth = (
            0.5 * (frozen["G_mm"] - frozen["C_mm"])
            / math.tan(math.radians(frozen["chamfer_angle_deg"]))
        )
        onset = (
            barrier_front
            + frozen["plug_B_to_socket_face_mm"]
            + frozen["lip_mm"]
            + chamfer_depth
        )
        interference = 0.5 * (frozen["C_mm"] - frozen["F_mm"])
        overlap_cap = 0.5 * (frozen["E_mm"] - frozen["F_mm"])
        post_contact_travel = frozen["bottoming_stop_separation_mm"] - onset
        tip_insertion = (
            frozen["bottoming_stop_separation_mm"]
            - frozen["plug_B_to_socket_face_mm"]
            - barrier_front
        )
        mouth_radius_m = frozen["G_mm"] * 0.0005
        throat_radius_m = frozen["F_mm"] * 0.0005
        barrier_tip_radius_m = frozen["C_mm"] * 0.0005
        barrier_base_radius_m = frozen["E_mm"] * 0.0005
        socket_face_m = frozen["plug_B_to_socket_face_mm"] * 0.001
        lip_end_m = (frozen["plug_B_to_socket_face_mm"] + frozen["lip_mm"]) * 0.001
        socket_chamfer_axial_mm = (
            0.5 * (frozen["G_mm"] - frozen["F_mm"])
            / math.tan(math.radians(frozen["chamfer_angle_deg"]))
        )
        chamfer_end_m = lip_end_m + socket_chamfer_axial_mm * 0.001
        barrier_front_m = barrier_front * 0.001
        barrier_tip_end_m = (
            barrier_front + variant_controls["invariant_nominal_r7_fields"][
                "barrier_tip_straight_axial_mm"
            ]
        ) * 0.001
        pin_end_m = frozen["N_mm"] * 0.001
        backing_hole_radius_m = barrier_base_radius_m
        insert_hole_vertex_radius_m = mouth_radius_m / math.cos(math.radians(7.5))
        backing_hole_vertex_radius_m = (
            backing_hole_radius_m / math.cos(math.radians(7.5))
        )
        socket_bands = []
        if frozen["lip_mm"] > 0.0:
            socket_bands.append({
                "profile_band_index": 0,
                "depth_start_m": socket_face_m,
                "depth_end_m": lip_end_m,
                "inner_radius_start_m": mouth_radius_m,
                "inner_radius_end_m": mouth_radius_m,
                "outer_radius_m": 0.00125,
            })
        socket_bands.extend([
            {
                "profile_band_index": 1,
                "depth_start_m": lip_end_m,
                "depth_end_m": chamfer_end_m,
                "inner_radius_start_m": mouth_radius_m,
                "inner_radius_end_m": throat_radius_m,
                "outer_radius_m": 0.00125,
            },
            {
                "profile_band_index": 2,
                "depth_start_m": chamfer_end_m,
                "depth_end_m": 0.002,
                "inner_radius_start_m": throat_radius_m,
                "inner_radius_end_m": throat_radius_m,
                "outer_radius_m": 0.00125,
            },
        ])
        entry_piece_count = len(socket_bands) * 24
        expected_authored = {
            "pins": {
                "local_z_interval_m": [0.01002, pin_end_m],
                "center_local_z_m": (0.01002 + pin_end_m) * 0.5,
                "height_m": pin_end_m - 0.01002,
            },
            "pin_barrier": {
                "axial_profile_bands": [
                    {
                        "profile_band_index": 0,
                        "depth_start_m": barrier_front_m,
                        "depth_end_m": barrier_tip_end_m,
                        "inner_radius_start_m": 0.000480,
                        "inner_radius_end_m": 0.000480,
                        "outer_radius_start_m": barrier_tip_radius_m,
                        "outer_radius_end_m": barrier_tip_radius_m,
                    },
                    {
                        "profile_band_index": 1,
                        "depth_start_m": barrier_tip_end_m,
                        "depth_end_m": pin_end_m,
                        "inner_radius_start_m": 0.000480,
                        "inner_radius_end_m": 0.000480,
                        "outer_radius_start_m": barrier_tip_radius_m,
                        "outer_radius_end_m": barrier_base_radius_m,
                    },
                ],
            },
            "socket_entry": {
                "active_profile_band_indices": [
                    row["profile_band_index"] for row in socket_bands
                ],
                "profile_bands": socket_bands,
                "expected_convex_piece_count_per_entry": entry_piece_count,
                "expected_total_convex_piece_count": entry_piece_count * 61,
            },
            "hard_insert_face": {
                "local_depth_interval_m": [socket_face_m, socket_face_m + 0.0005],
                "required_hole_clear_radius_m": mouth_radius_m,
                "hole_polygon_vertex_radius_m": insert_hole_vertex_radius_m,
            },
            "fixed_backing_face": {
                "local_z_interval_m": [pin_end_m, pin_end_m + 0.0005],
                "required_hole_clear_radius_m": backing_hole_radius_m,
                "hole_polygon_vertex_radius_m": backing_hole_vertex_radius_m,
            },
            "plug_metal_stop": {
                "local_depth_interval_m": [
                    frozen["bottoming_stop_separation_mm"] * 0.001,
                    frozen["bottoming_stop_separation_mm"] * 0.001 + 0.0003,
                ],
            },
        }
        _require_exact_numeric_tree(
            variant.get("authored_collider_values"), expected_authored,
            f"size-20 variant {name}.authored_collider_values",
        )
        for field, expected in (
            ("derived_first_contact_separation_mm", onset),
            ("derived_post_contact_axial_travel_mm", post_contact_travel),
            ("derived_tip_insertion_at_bottoming_mm", tip_insertion),
            ("derived_tip_throat_radial_interference_mm", interference),
            ("unreached_E_to_F_geometric_overlap_cap_mm", overlap_cap),
        ):
            if not math.isclose(
                _finite(variant.get(field), f"{name}.{field}"),
                expected,
                abs_tol=1.0e-6,
            ):
                raise ValueError(f"size-20 variant {name}.{field} is inconsistent")
        expected_variant_fields = set(frozen) | {
            "authored_collider_values",
            "derived_first_contact_separation_mm",
            "derived_post_contact_axial_travel_mm",
            "derived_tip_insertion_at_bottoming_mm",
            "derived_tip_throat_radial_interference_mm",
            "unreached_E_to_F_geometric_overlap_cap_mm",
        }
        if set(variant) != expected_variant_fields:
            raise ValueError(f"size-20 variant {name} field inventory changed")


def _range_exact(
    parent: Mapping[str, Any],
    field: str,
    minimum: float,
    maximum: float,
    label: str,
) -> Mapping[str, Any]:
    values = _mapping(parent.get(field), f"{label}.{field}")
    if (
        _finite(values.get("min"), f"{label}.{field}.min") != minimum
        or _finite(values.get("max"), f"{label}.{field}.max") != maximum
    ):
        raise ValueError(f"{label}.{field} changed from the frozen range")
    return values


def _validate_axial_interface(geometry: Mapping[str, Any]) -> None:
    axial = _mapping(geometry.get("axial_interface"), "public_geometry.axial_interface")
    if axial.get("source_basis") != (
        "PUBLIC_SPEC_VERIFIED_AND_DERIVED_WITH_EXPLICIT_SIMULATION_PROXIES"
    ):
        raise ValueError("axial interface source boundary changed")
    axes = _mapping(axial.get("local_depth_axes"), "axial local depth axes")
    if axes != {
        "receptacle": "datum_B_toward_receptacle_interior_is_positive",
        "plug": "datum_B_toward_plug_interior_is_positive",
        "datum_B_definition": "each_connector_shell_front_face",
        "datum_B_planes_coincident_at_full_mate": False,
    }:
        raise ValueError("the two local datum-B axes changed")
    pair = _mapping(
        axial.get("current_pair_contact_polarity"), "current pair polarity"
    )
    if (
        pair.get("fixed_receptacle") != "pin"
        or pair.get("loose_plug") != "socket"
        or pair.get("applicable_receptacle_dimensions")
        != ["A", "M", "N", "pin_barrier_front_unlettered"]
        or pair.get("applicable_plug_dimensions")
        != ["L_internal_alias", "Y", "Z"]
    ):
        raise ValueError("current /20 pin to /26 socket axial mapping changed")

    receptacle = _mapping(
        axial.get("receptacle_from_B_depth_mm"), "receptacle axial dimensions"
    )
    for field, minimum, maximum in (
        ("A_static_seal_first_touch", 14.50, 14.73),
        ("M_pin_tip", 9.50, 10.54),
        ("N_pin_insert_face", 15.04, 15.46),
        ("pin_barrier_front_unlettered", 13.72, 14.05),
        ("derived_pin_exposed_length", 4.50, 5.96),
        ("derived_barrier_projection", 0.99, 1.74),
    ):
        values = _range_exact(
            receptacle, field, minimum, maximum, "receptacle axial dimensions"
        )
        expected_kind = (
            "PUBLIC_SPEC_DERIVED" if field.startswith("derived_")
            else "PUBLIC_SPEC_VERIFIED"
        )
        if values.get("source_kind") != expected_kind:
            raise ValueError(f"{field} source classification changed")
    receptacle_nominal = _mapping(
        receptacle.get("nominal_for_r7"), "receptacle nominal axial dimensions"
    )
    expected_receptacle_nominal = {
        "A_static_seal_first_touch": 14.615,
        "M_pin_tip": 10.02,
        "N_pin_insert_face": 15.25,
        "pin_barrier_front_unlettered": 13.93,
        "pin_barrier_projection_from_insert_face": 1.32,
    }
    if receptacle_nominal != expected_receptacle_nominal:
        raise ValueError("receptacle nominal axial dimensions changed")

    plug = _mapping(axial.get("plug_control_chain_mm"), "plug axial dimensions")
    _range_exact(
        plug,
        "L_B_to_common_rear_control_plane",
        15.01,
        15.09,
        "plug axial dimensions",
    )
    y = _mapping(
        plug.get("Y_socket_spring_touch_to_rear_plane"),
        "plug axial dimensions.Y",
    )
    if _finite(y.get("min"), "plug Y minimum") != 12.45 or y.get("max") is not None:
        raise ValueError("plug Y must remain a one-sided 12.45 mm minimum")
    _range_exact(
        plug,
        "Z_socket_insert_face_to_rear_plane",
        14.76,
        15.06,
        "plug axial dimensions",
    )
    _range_exact(
        plug,
        "derived_B_to_socket_insert_face",
        -0.05,
        0.33,
        "plug axial dimensions",
    )
    if _finite(
        _mapping(
            plug.get("derived_B_to_socket_spring_touch_upper_bound"),
            "plug socket spring upper bound",
        ).get("max"),
        "plug socket spring upper bound",
    ) != 2.64:
        raise ValueError("plug socket spring datum-B upper bound changed")
    if _finite(
        _mapping(
            plug.get("derived_socket_spring_depth_from_insert_upper_bound"),
            "plug socket depth upper bound",
        ).get("max"),
        "plug socket depth upper bound",
    ) != 2.61:
        raise ValueError("plug socket spring depth upper bound changed")
    plug_nominal = _mapping(plug.get("nominal_for_r7"), "plug nominal axial chain")
    required_nominal = {
        "L_B_to_common_rear_control_plane": 15.05,
        "Z_socket_insert_face_to_rear_plane": 14.91,
        "B_to_socket_insert_face": 0.14,
        "B_to_socket_spring_touch": 2.00,
        "Y_socket_spring_touch_to_rear_plane": 13.05,
        "socket_spring_depth_from_insert_face": 1.86,
        "nominal_source_kind": (
            "FROZEN_SIMULATION_PROXY_WITHIN_PUBLIC_ONE_SIDED_LIMITS"
        ),
    }
    if plug_nominal != required_nominal:
        raise ValueError("plug nominal Y/Z/L chain changed")
    if not math.isclose(
        plug_nominal["L_B_to_common_rear_control_plane"]
        - plug_nominal["Y_socket_spring_touch_to_rear_plane"],
        plug_nominal["B_to_socket_spring_touch"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("plug nominal Y was treated as a direct datum-B coordinate")

    bottoming = _mapping(axial.get("metal_bottoming"), "metal bottoming")
    if (
        bottoming.get("definition")
        != "physical_receptacle_engaging_shell_to_plug_internal_engaging_shell_contact"
        or bottoming.get("public_spec_confirms_surface_pair_but_not_their_datum_assignment")
        is not True
        or bottoming.get("calibration_range_is_not_a_public_bottoming_depth_claim")
        is not True
        or bottoming.get("determined_by_physical_collision_not_pose_or_boolean") is not True
    ):
        raise ValueError("metal bottoming must remain a physical shell stop")
    receptacle_stop = _mapping(
        bottoming.get("receptacle_stop_surface"), "receptacle stop surface"
    )
    plug_stop = _mapping(
        bottoming.get("plug_internal_stop_surface"), "plug stop surface"
    )
    if (
        _finite(
            receptacle_stop.get("local_depth_from_receptacle_B_mm"),
            "receptacle stop depth",
        )
        != 0.0
        or receptacle_stop.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or _finite(
            plug_stop.get("local_depth_from_plug_B_mm"), "plug stop depth"
        )
        != 15.05
        or plug_stop.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("both bottoming stop assignments must remain explicit proxies")
    bottoming_range = _mapping(
        bottoming.get("bottoming_proxy_calibration_range_mm"),
        "bottoming proxy calibration range",
    )
    if {
        key: _finite(bottoming_range.get(key), f"bottoming range {key}")
        for key in ("min", "nominal", "max")
    } != {"min": 15.01, "nominal": 15.05, "max": 15.09}:
        raise ValueError("bottoming proxy calibration range changed")
    if _finite(
        bottoming.get("full_mate_datum_B_separation_mm"), "full mate B separation"
    ) != 15.05:
        raise ValueError("full-mate datum-B separation changed")
    stack = _mapping(bottoming.get("nominal_stack_mm"), "nominal bottoming stack")
    expected_stack = {
        "seal_first_touch_separation": 14.615,
        "seal_physical_deflection_at_bottoming": 0.435,
        "first_electrical_touch_separation": 12.02,
        "electrical_engagement_at_bottoming": 3.03,
        "pin_barrier_first_touch_from_collider_bounds": 14.305,
        "pin_barrier_post_contact_axial_travel_to_bottoming": 0.745,
        "pin_barrier_nominal_tip_to_throat_radial_interference": 0.295,
        "plug_socket_insert_world_depth_from_receptacle_B": 14.91,
        "receptacle_pin_insert_depth_from_receptacle_B": 15.25,
    }
    if {
        key: _finite(stack.get(key), f"nominal stack {key}")
        for key in expected_stack
    } != expected_stack:
        raise ValueError("nominal bottoming/contact stack changed")
    if not math.isclose(
        stack["seal_physical_deflection_at_bottoming"],
        bottoming["full_mate_datum_B_separation_mm"]
        - stack["seal_first_touch_separation"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("seal deflection is inconsistent with the bottoming stack")
    if not math.isclose(
        stack["electrical_engagement_at_bottoming"],
        bottoming["full_mate_datum_B_separation_mm"]
        - stack["first_electrical_touch_separation"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("electrical engagement is inconsistent with bottoming")
    if not math.isclose(
        receptacle_nominal["N_pin_insert_face"]
        - receptacle_nominal["pin_barrier_front_unlettered"],
        receptacle_nominal["pin_barrier_projection_from_insert_face"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("Figure-14 nominal pin-barrier projection is inconsistent")
    size20_detail = _mapping(
        _mapping(
            geometry.get("contact_pattern_25_61"), "25-61 contact pattern"
        ).get("series_III_size20_interface_detail"),
        "Series-III size-20 interface detail",
    )
    interface_blueprint = _mapping(
        size20_detail.get("r7_collision_blueprint"), "size-20 collision blueprint"
    )
    if not math.isclose(
        interface_blueprint["nominal_first_contact_datum_B_separation_mm"],
        stack["pin_barrier_first_touch_from_collider_bounds"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        bottoming["full_mate_datum_B_separation_mm"]
        - stack["pin_barrier_first_touch_from_collider_bounds"],
        stack["pin_barrier_post_contact_axial_travel_to_bottoming"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        0.5
        * (
            size20_detail["pin_barrier_C_diameter_mm"]["nominal"]
            - size20_detail["socket_entry_F_diameter_mm"]["nominal"]
        ),
        stack["pin_barrier_nominal_tip_to_throat_radial_interference"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("pin-barrier collider onset/deflection chain is inconsistent")

    thread_axial = _mapping(
        axial.get("thread_axial_controls"), "thread axial controls"
    )
    _range_exact(
        thread_axial,
        "B_to_figure3_THREAD_control_plane_mm",
        8.33,
        9.91,
        "thread axial controls",
    )
    if _finite(
        _mapping(
            thread_axial.get("B_to_figure3_THREAD_control_plane_mm"),
            "thread control plane",
        ).get("nominal"),
        "thread control plane nominal",
    ) != 9.12:
        raise ValueError("thread control-plane nominal changed")
    for field, expected in (
        ("B_to_second_unnamed_control_plane_min_mm", 11.79),
        ("alternate_view_full_thread_axial_length_min_mm", 10.87),
        ("plug_DD_entry_forward_extent_min_mm", 2.16),
        ("figure3_leftmost_control_plane_from_B_max_mm", 16.92),
    ):
        if _finite(thread_axial.get(field), f"thread axial controls.{field}") != expected:
            raise ValueError(f"thread axial controls.{field} changed")
    if thread_axial.get("exact_unnamed_endpoint_function_claimed") is not False:
        raise ValueError("unnamed Figure-3 thread endpoints cannot gain invented names")
    alternate = _mapping(
        thread_axial.get("alternate_view_applicability"),
        "thread alternate-view applicability",
    )
    if (
        alternate.get(
            "current_normal_view_r7_proxy_uses_10p87_as_geometry_constraint"
        )
        is not False
        or alternate.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or not isinstance(alternate.get("decision"), str)
    ):
        raise ValueError("alternate-view thread applicability was not frozen closed")
    one_sided = _mapping(
        thread_axial.get("one_sided_r7_proxy_values_mm"),
        "thread one-sided proxy values",
    )
    if (
        _finite(one_sided.get("second_unnamed_control_plane_from_B"), "second thread plane")
        != 11.79
        or _finite(
            one_sided.get("alternate_full_thread_min_reference"),
            "alternate full-thread reference",
        )
        != 10.87
        or one_sided.get("alternate_full_thread_used_by_r7") is not False
        or _finite(one_sided.get("plug_DD_entry_forward_extent"), "plug DD extent")
        != 2.16
        or one_sided.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("one-sided Figure-3 values lack exact r7 proxies")
    proxy = _mapping(
        thread_axial.get("r7_force_transmitting_geometry"),
        "r7 force-transmitting thread geometry",
    )
    interval = _mapping(
        proxy.get("receptacle_loaded_rail_interval_from_B_mm"),
        "r7 rail interval",
    )
    rail_start = _finite(interval.get("start"), "r7 rail start")
    rail_end = _finite(interval.get("end"), "r7 rail end")
    expected_proxy = {
        "nominal_loaded_entry_control_plane_from_B_mm": 9.12,
        "nominal_full_mate_follower_plane_from_B_mm": 16.74,
        "nominal_active_axial_travel_mm": 7.62,
        "plug_follower_signed_depth_from_plug_B_mm": -1.69,
        "nominal_thread_entry_datum_B_separation_mm": 7.43,
        "nominal_bottoming_datum_B_separation_mm": 15.05,
    }
    if (rail_start, rail_end) != (9.12, 16.74) or {
        key: _finite(proxy.get(key), f"thread proxy {key}")
        for key in expected_proxy
    } != expected_proxy:
        raise ValueError("r7 physical thread interval or follower geometry changed")
    if (
        proxy.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or proxy.get("all_three_rails_share_identical_lead_and_120deg_phase")
        is not True
    ):
        raise ValueError("r7 physical thread policy changed")
    if not math.isclose(rail_end - rail_start, 7.62, abs_tol=1.0e-12):
        raise ValueError("r7 loaded rail is not one exact lead")
    if rail_end > 16.92:
        raise ValueError("r7 rail exceeds the Figure-3 control-plane cap")
    if not rail_start == 9.12 <= 11.79 <= rail_end:
        raise ValueError("r7 rail does not contain both selected control planes")
    if not math.isclose(16.74 - 9.12, 7.62, abs_tol=1.0e-12):
        raise ValueError("r7 loaded thread path is not one exact lead")
    if not math.isclose(15.05 - 7.43, 7.62, abs_tol=1.0e-12):
        raise ValueError("r7 thread entry-to-bottoming path is not one revolution")
    if abs(-1.69) > 2.16:
        raise ValueError("r7 follower lies outside the selected plug DD extent")
    red_band = _mapping(axial.get("red_band_coverage"), "red band coverage")
    _range_exact(red_band, "figure3_span_mm", 13.20, 13.74, "red band coverage")
    if (
        red_band.get("exact_left_endpoint_name_claimed") is not False
        or red_band.get("physical_pass_gate") is not False
    ):
        raise ValueError("the red band cannot become physical truth or a pass gate")


def _validate_components(document: Mapping[str, Any]) -> None:
    raw = document.get("component_completeness")
    if not isinstance(raw, list):
        raise ValueError("component_completeness must be a list")
    components = {}
    for index, value in enumerate(raw):
        component = _mapping(value, f"component_completeness[{index}]")
        component_id = component.get("id")
        if not isinstance(component_id, str) or component_id in components:
            raise ValueError("component ids must be unique text")
        components[component_id] = component
        target = component.get("a5_representation")
        if component.get("affects_force_or_motion") is True and (
            not isinstance(target, str) or not target.startswith("physical_")
        ):
            raise ValueError(
                f"force-bearing component {component_id} lacks a physical A5 target"
            )
        acceptance = component.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance:
            raise ValueError(f"component {component_id} lacks acceptance benches")
        if any(item not in REQUIRED_BENCH_IDS for item in acceptance):
            raise ValueError(f"component {component_id} names an unknown bench")
    if frozenset(components) != REQUIRED_COMPONENT_IDS:
        missing = sorted(REQUIRED_COMPONENT_IDS - frozenset(components))
        extra = sorted(frozenset(components) - REQUIRED_COMPONENT_IDS)
        raise ValueError(f"component inventory differs: missing={missing}, extra={extra}")


def _validate_a2_collision_authoring_blueprint(document: Mapping[str, Any]) -> None:
    blueprint = _mapping(
        document.get("a2_collision_authoring_blueprint"),
        "a2_collision_authoring_blueprint",
    )
    required_sections = {
        "status", "source_kind", "authoring_rule", "global", "rigid_body_owners",
        "explicit_convex_piece_contract", "canonical_primitive_recipes",
        "offset_classes", "response_contract",
        "contact_layout", "connector_shells_and_keying", "thread",
        "electrical_contacts", "pin_barriers", "spring_fingers",
        "peripheral_seal", "anti_decoupling_detent", "metal_bottoming",
        "nut_body_shoulders", "filtering",
    }
    if set(blueprint) != required_sections:
        raise ValueError("A2 collision-authoring blueprint section inventory changed")
    if (
        blueprint.get("status")
        != "FROZEN_FOR_MECHANICAL_A2_AUTHORING"
        or blueprint.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or blueprint.get("authoring_rule")
        != "contract_is_the_only_numeric_geometry_source_and_A2_may_not_choose_values"
    ):
        raise ValueError("A2 collision blueprint release boundary changed")
    global_rules = _mapping(blueprint.get("global"), "collision blueprint global")
    expected_global = {
        "physics_backend": "PhysX",
        "simulation_device": "cpu",
        "gpu_compatible_explicit_convex_budget_required": False,
        "canonical_readback_pose": {
            "fixed_receptacle_translation_m": [0.0, 0.0, 0.0],
            "fixed_receptacle_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "loose_plug_translation_m": [0.0, 0.0, 0.01505],
            "loose_plug_rotation_wxyz": [0.0, 1.0, 0.0, 0.0],
            "body_assembly_child_transform_is_identity": True,
            "coupling_nut_child_transform_is_identity": True,
        },
        "geometry_abs_tolerance_m": 0.00000001,
        "angle_abs_tolerance_deg": 0.000001,
        "missing_count_allowed": 0,
        "unexpected_count_allowed": 0,
        "duplicate_count_allowed": 0,
        "negative_or_mirrored_scale_allowed": False,
        "time_sampled_collision_geometry_allowed": False,
        "automatic_convex_decomposition_allowed": False,
        "triangle_mesh_approximation_allowed_on_dynamic_owner": False,
        "sdf_allowed": False,
        "primitive_or_explicit_convexHull_only": True,
        "all_connector_collision_offset_class": "fine_connector",
        "automatic_contact_or_rest_offset_allowed": False,
    }
    if dict(global_rules) != expected_global:
        raise ValueError("global collider authoring/cooking rules changed")
    if blueprint.get("rigid_body_owners") != {
        "FixedReceptacle": {
            "prim_path": f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle",
            "rigidBodyEnabled": True,
            "kinematicEnabled": False,
            "motion_class": "dynamic_world_fixed_by_enabled_fixed_joint_chain_in_tabletop_scene",
        },
        "BodyAssembly": {
            "prim_path": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly",
            "rigidBodyEnabled": True,
            "kinematicEnabled": False,
            "motion_class": "dynamic",
        },
        "CouplingNut": {
            "prim_path": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut",
            "rigidBodyEnabled": True,
            "kinematicEnabled": False,
            "motion_class": "dynamic",
        },
    }:
        raise ValueError("connector rigid-owner identities changed")
    convex = _mapping(
        blueprint.get("explicit_convex_piece_contract"), "explicit convex contract"
    )
    if dict(convex) != {
        "typeName": "Mesh",
        "physics_approximation": "convexHull",
        "exact_topology_must_be_authored": True,
        "automatic_convex_decomposition_allowed": False,
        "subdivisionScheme": "none",
        "orientation": "rightHanded",
        "closed_manifold_required": True,
        "positive_volume_required": True,
        "convexity_required": True,
        "convex_hull_surface_error_m_max": 0.00000001,
        "adjacent_seam_vertices_shared_exactly": True,
        "angular_gap_m_max": 0.00000001,
        "angular_overlap_m_max": 0.00000001,
        "vertices_per_piece_max": 64,
        "polygons_per_piece_max": 64,
        "vertices_per_face_max": 32,
        "required_generator_fields": [
            "profile_id", "instance_index", "piece_index",
            "angular_segment_index", "profile_band_index", "theta0_deg",
            "theta1_deg", "phase_origin_deg", "exact_points",
            "exact_faceVertexCounts", "exact_faceVertexIndices",
        ],
    }:
        raise ValueError("explicit convex piece topology contract changed")
    recipes = _mapping(
        blueprint.get("canonical_primitive_recipes"), "canonical primitive recipes"
    )
    if set(recipes) != {
        "coordinate_and_rounding", "angular_interval_convention",
        "annular_wedge_8v_v1", "tangent_profile_prism_v1",
        "annular_wedge_corner_templates",
        "radial_tangent_box_8v_v1", "planar_triangle_z_prism_6v_v1",
        "thread_rail_hexahedron_8v_v1",
        "keyway_pslg_v1",
    }:
        raise ValueError("canonical primitive recipe inventory changed")
    if (
        _mapping(recipes.get("coordinate_and_rounding"), "recipe rounding").get(
            "quantization_rule"
        )
        != "round_to_nearest_ties_away_from_zero_after_complete_analytic_point_evaluation"
        or _mapping(recipes.get("annular_wedge_8v_v1"), "annular recipe").get(
            "faceVertexIndices"
        ) != [
            3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4,
            1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7,
        ]
        or _mapping(recipes.get("annular_wedge_8v_v1"), "annular recipe").get(
            "exact_corner_radius_input_order"
        ) != [
            "inner_theta0_z0", "outer_theta0_z0", "outer_theta1_z0",
            "inner_theta1_z0", "inner_theta0_z1", "outer_theta0_z1",
            "outer_theta1_z1", "inner_theta1_z1",
        ]
        or _mapping(recipes.get("annular_wedge_8v_v1"), "annular recipe").get(
            "every_corner_radius_is_an_explicit_family_value_or_formula"
        ) is not True
        or _mapping(recipes.get("tangent_profile_prism_v1"), "profile recipe").get(
            "automatic_convex_hull_vertex_deletion_allowed"
        )
        is not False
        or _mapping(recipes.get("keyway_pslg_v1"), "keyway recipe").get(
            "slot_outer_end_geometry"
        )
        != "straight_chord_between_the_two_side_intersections_with_slot_outer_end_radius"
    ):
        raise ValueError("canonical primitive point/topology recipe changed")
    _require_exact_fields(
        _mapping(
            recipes.get("thread_rail_hexahedron_8v_v1"),
            "thread rail primitive recipe",
        ),
        {
            "theta0_formula_deg": "start_phase_deg+segment_index*1deg",
            "theta1_formula_deg": "start_phase_deg+(segment_index+1)*1deg",
            "theta_center_formula_deg":
            "start_phase_deg+(segment_index+0.5)*1deg",
            "contact_z0_formula_m":
            "0.00912+0.00762*segment_index/360",
            "contact_z1_formula_m":
            "0.00912+0.00762*(segment_index+1)/360",
            "final_segment_endpoint_policy":
            "segment_359_z1_is_exactly_0.01674_and_theta_endpoint_is_not_wrapped_for_z",
            "solid_z0_formula_m": "contact_z0_m+0.00030",
            "solid_z1_formula_m": "contact_z1_m+0.00030",
            "expected_per_segment_contact_z_rise_m":
            0.000021166666666666667,
            "all_360_contact_z_steps_are_strictly_positive": True,
            "seam_z_gap_and_overlap_m": 0.0,
            "contact_surface_nonplanar_quad_forbidden": True,
            "contact_surface_diagonal": [0, 2],
            "solid_surface_diagonal": [5, 7],
            "faceVertexCounts": [3, 3, 3, 3, 4, 4, 4, 4],
            "faceVertexIndices": [
                3, 2, 0, 2, 1, 0, 4, 5, 7, 5, 6, 7,
                0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
                3, 0, 4, 7,
            ],
        },
        "thread rail primitive recipe",
    )
    if recipes.get("annular_wedge_corner_templates") != {
        "constant_direct_endpoint_interval_v1": {
            "corner_radius_vector": [
                "inner", "outer", "outer", "inner",
                "inner", "outer", "outer", "inner",
            ],
            "specified_inner_and_outer_radii_are_vertex_radii_not_chord_tangent_radii": True,
        },
        "axial_band_direct_endpoint_profile_v1": {
            "corner_radius_vector": [
                "inner_z0", "outer_z0", "outer_z0", "inner_z0",
                "inner_z1", "outer_z1", "outer_z1", "inner_z1",
            ],
            "specified_profile_radii_are_vertex_radii_not_chord_tangent_radii": True,
        },
        "clearance_preserving_inner_chord_outer_envelope_v1": {
            "corner_radius_vector": [
                "inner/cos(half_step)", "outer", "outer",
                "inner/cos(half_step)", "inner/cos(half_step)",
                "outer", "outer", "inner/cos(half_step)",
            ],
            "inner_chord_is_tangent_to_required_clear_radius": True,
            "outer_vertices_do_not_exceed_required_outer_radius": True,
        },
        "circumferential_profile_explicit_corners_v1": {
            "corner_radius_vector": [
                "inner_theta0_z0", "outer_theta0_z0", "outer_theta1_z0",
                "inner_theta1_z0", "inner_theta0_z1", "outer_theta0_z1",
                "outer_theta1_z1", "inner_theta1_z1",
            ],
            "all_eight_corner_radii_must_be_supplied_by_family_formula": True,
        },
    }:
        raise ValueError("annular-wedge corner templates changed")
    if blueprint.get("offset_classes") != {
        "fine_connector": {"contactOffset_m": 0.00001, "restOffset_m": 0.0},
        "general": {"contactOffset_m": 0.00005, "restOffset_m": 0.0},
    }:
        raise ValueError("A2 collision offset classes changed")
    response = _mapping(blueprint.get("response_contract"), "response contract")
    _require_exact_fields(
        response,
        {
            "semantic_material_attribute": "kcg:materialRole",
            "physical_response_attribute": "kcg:responseRole",
            "exactly_one_semantic_material_role_per_collider": True,
            "exactly_one_response_role_per_collider": True,
            "compliant_pair_exactly_one_compliant_side_required": True,
            "hard_side_resolved_compliant_stiffness_n_m": 0.0,
            "hard_side_resolved_compliant_damping_n_s_m": 0.0,
            "compliant_side_required_schemas": ["PhysicsMaterialAPI", "PhysxMaterialAPI"],
            "compliant_side_stiffness_must_be_positive": True,
            "compliant_side_damping_must_be_nonnegative": True,
            "compliant_side_accelerationSpring": False,
            "contactOffset_is_not_physical_deflection": True,
            "signed_distance_formula": "s_rest=s_geom-(restOffset_A+restOffset_B)",
            "overlap_formula": "overlap=max(0,-s_rest)",
            "hard_or_compliant_classification_requires_resolved_pair_and_effective_binding": True,
        },
        "response contract",
    )
    layout = _mapping(blueprint.get("contact_layout"), "contact layout")
    if layout != {
        "coordinate_source": "src/kcg_connector/config/d38999_keyed_public_spec_v2.yaml",
        "arrangement": "25-61",
        "label_order": "exact_source_order",
        "count": 61,
        "positions_in_exactly": [list(row) for row in EXPECTED_CONTACT_POSITIONS_IN],
        "unit_conversion": "source_inches_to_stage_meters_exact_0p0254",
        "fixed_receptacle_view": "front_face_source_coordinates",
        "loose_plug_view": "mirror_exactly_once_through_canonical_mating_transform",
        "fixed_contact_local_xy_formula_m": ["source_x_in*0.0254", "source_y_in*0.0254"],
        "plug_contact_local_xy_formula_m": ["source_x_in*0.0254", "-source_y_in*0.0254"],
        "canonical_plug_rotation_maps_local_xyz_to_world": ["x", "-y", "-z"],
        "canonical_same_label_world_xy_error_m": 0.0,
        "controller_or_filter_may_not_use_same_label_to_enable_contact": True,
    }:
        raise ValueError("61-contact layout authoring contract changed")

    shells = _mapping(
        blueprint.get("connector_shells_and_keying"), "connector shells/keying"
    )
    if set(shells) != {
        "source_kind", "annular_wedge_recipe", "receptacle_keyway_shell",
        "receptacle_thread_carrier", "plug_mating_shell", "plug_keys",
        "polarization_event", "deterministic_PSLG_contract", "coupling_nut",
        "body_assembly_rear_body",
    }:
        raise ValueError("shell/key/nut blueprint inventory changed")
    annulus = _mapping(shells.get("annular_wedge_recipe"), "annular wedge recipe")
    _require_exact_fields(
        annulus,
        {
            "corner_template_id":
            "clearance_preserving_inner_chord_outer_envelope_v1",
            "angular_segment_count": 360,
            "angular_step_deg": 1.0,
            "phase_origin_deg": 0.5,
            "inner_endpoint_radius_formula": "required_inner_radius/cos(0.5deg)",
            "outer_endpoint_radius_formula": "required_outer_radius",
            "inner_chord_is_tangent_to_required_clear_bore": True,
            "outer_polygon_does_not_exceed_required_outer_radius": True,
            "shared_edges_zero_gap_zero_overlap": True,
            "physics_approximation": "convexHull",
            "capped_solid_cylinder_allowed": False,
        },
        "annular wedge recipe",
    )
    keyway = _mapping(shells.get("receptacle_keyway_shell"), "receptacle keyway shell")
    _require_exact_fields(
        keyway,
        {
            "owner": "FixedReceptacle",
            "local_z_interval_m": [0.0, 0.00900],
            "outer_radius_m": 0.019610,
            "clear_bore_radius_m": 0.0179575,
            "keyway_slot_end_radius_m": 0.019030,
            "keyway_center_angles_deg": [0.0, 80.0, 142.0, 196.0, 293.0],
            "keyway_parallel_wall_widths_m": [0.00320, 0.00160, 0.00160, 0.00160, 0.00160],
            "keyway_void_sidewalls_are_exact_parallel_lines": True,
            "deleted_one_degree_sector_mask_allowed": False,
            "one_unslotted_annulus_layer_may_not_overlap_this_shell": True,
            "response_role": "hard_receptacle_keyway_shell",
        },
        "receptacle keyway shell",
    )
    plug_shell = _mapping(shells.get("plug_mating_shell"), "plug mating shell")
    _require_exact_fields(
        plug_shell,
        {
            "owner": "BodyAssembly",
            "local_depth_interval_m": [0.0, 0.01505],
            "radial_interval_m": [0.0170825, 0.017845],
            "nominal_radial_clearance_to_receptacle_bore_m": 0.0001125,
            "response_role": "hard_plug_mating_shell",
        },
        "plug mating shell",
    )
    keys = _mapping(shells.get("plug_keys"), "plug keys")
    _require_exact_fields(
        keys,
        {
            "owner": "BodyAssembly", "count": 5,
            "center_angles_deg": [0.0, -80.0, -142.0, -196.0, -293.0],
            "canonical_world_center_angles_deg": [0.0, 80.0, 142.0, 196.0, 293.0],
            "tangent_frame_widths_m": [0.00254, 0.00132, 0.00132, 0.00132, 0.00132],
            "radial_interval_m": [0.017845, 0.018855],
            "local_depth_interval_m": [0.00650, 0.01374],
            "exact_axial_length_m": 0.00724,
            "leading_nose_axial_length_m": 0.00018,
            "leading_nose_profile": "linear_radial_ramp",
            "leading_nose_angle_from_local_plus_z_deg": 79.89496363496205,
            "leading_nose_angle_is_derived_from_radial_rise_over_axial_length": True,
            "leading_tip_bound_local_minZ_m": 0.00650,
            "full_radial_section_starts_local_z_m": 0.00668,
            "parallel_tangent_sidewalls_required": True,
            "each_key_is_two_explicit_convex_prisms_nose_and_full_section": True,
            "response_role": "hard_polarizing_key",
        },
        "plug keys",
    )
    if any(
        key_width >= slot_width
        for key_width, slot_width in zip(
            keys["tangent_frame_widths_m"], keyway["keyway_parallel_wall_widths_m"]
        )
    ):
        raise ValueError("key nominal width must remain below its keyway width")
    polarization = _mapping(shells.get("polarization_event"), "polarization event")
    if (
        polarization.get("derived_correct_yaw_event_separation_m") != 0.00650
        or polarization.get("wrong_yaw_hard_blocking_begins_no_later_than_separation_m")
        != 0.00668
        or polarization.get("event_is_not_a_contact_or_runtime_boolean") is not True
        or polarization.get("all_five_keys_to_entire_keyway_shell_pairs_enabled") is not True
        or polarization.get("corresponding_key_index_filtering_allowed") is not False
        or polarization["wrong_yaw_hard_blocking_begins_no_later_than_separation_m"]
        >= 0.00743
    ):
        raise ValueError("physical five-key polarization event changed")
    pslg = _mapping(shells.get("deterministic_PSLG_contract"), "deterministic PSLG")
    if (
        pslg.get("exact_integer_orientation_and_incircle_predicates") is not True
        or pslg.get("Steiner_vertices_allowed") is not False
        or pslg.get("constrained_Delaunay_required") is not True
        or pslg.get("cocircular_tie_break") != "lexicographic_vertex_id"
        or pslg.get("each_triangle_extruded_as_one_closed_convex_prism") is not True
        or pslg.get("runtime_mesh_repair_or_tolerance_hole_closure_allowed") is not False
    ):
        raise ValueError("deterministic keyway PSLG/CDT contract changed")
    thread_carrier = _mapping(
        shells.get("receptacle_thread_carrier"), "receptacle thread carrier"
    )
    if (
        thread_carrier.get("local_z_interval_m") != [0.00900, 0.01692]
        or thread_carrier.get("radial_interval_m") != [0.0179575, 0.0199168]
        or thread_carrier.get("outer_radius_matches_thread_rail_inner_radius")
        is not True
    ):
        raise ValueError("receptacle thread-carrier support changed")
    nut = _mapping(shells.get("coupling_nut"), "coupling nut envelope")
    _require_exact_fields(
        nut,
        {
            "primitive_recipe_id": "annular_wedge_8v_v1",
            "corner_template_id": "constant_direct_endpoint_interval_v1",
            "owner": "CouplingNut", "wedge_segment_count": 96,
            "front_thread_carrier_local_z_interval_m": [-0.00200, 0.00900],
            "front_thread_carrier_radial_interval_m": [0.02090, 0.02400],
            "rear_grip_local_z_interval_m": [0.00900, 0.03050],
            "rear_grip_radial_interval_m": [0.02260, 0.02400],
            "robot_grip_target_local_z_interval_m": [0.00900, 0.02900],
            "capped_solid_cylinder_allowed": False,
            "response_role": "hard_coupling_nut",
        },
        "coupling nut envelope",
    )
    rear = _mapping(shells.get("body_assembly_rear_body"), "plug rear body")
    if (
        rear.get("primitive_recipe_id") != "annular_wedge_8v_v1"
        or rear.get("corner_template_id")
        != "axial_band_direct_endpoint_profile_v1"
        or rear.get("local_z_profile_bands") != [
            {"profile_band_index": 0, "z_start_m": 0.01505, "z_end_m": 0.01955,
             "inner_radius_m": 0.01710, "outer_radius_m": 0.02220},
            {"profile_band_index": 1, "z_start_m": 0.01955, "z_end_m": 0.02045,
             "inner_radius_m": 0.01710, "outer_radius_m": 0.02150},
            {"profile_band_index": 2, "z_start_m": 0.02045, "z_end_m": 0.03050,
             "inner_radius_m": 0.01710, "outer_radius_m": 0.02220},
        ]
        or rear.get("ordinary_rear_body_to_nut_annulus_pairs_filtered") is not True
    ):
        raise ValueError("plug rear-body detent relief or internal filtering changed")

    thread = _mapping(blueprint.get("thread"), "thread authoring blueprint")
    _require_exact_fields(
        thread,
        {
            "source_kind": "FROZEN_SIMULATION_PROXY",
            "rail_owner": "FixedReceptacle", "follower_owner": "CouplingNut",
            "start_count": 3, "start_phases_deg": [0.0, 120.0, 240.0],
            "segments_per_start": 360, "segment_angle_deg": 1.0,
            "rail_piece_shape": "explicit_convex_hexahedron",
            "contact_surface_equation_mm": "z=9.12+7.62*wrap(theta_deg-phase_deg,0,360)/360",
            "axial_interval_m": [0.00912, 0.01674], "pitch_radius_m": 0.0201168,
            "rail_radial_interval_m": [0.0199168, 0.0203168],
            "follower_radial_interval_m": [0.0199668, 0.0202668],
            "rail_axial_thickness_m": 0.00030,
            "follower_contact_surface_local_z_m": -0.00169,
            "follower_solid_interval_local_z_m": [-0.00169, -0.00139],
            "follower_tangential_width_m": 0.00020,
            "follower_center_angle_offset_from_start_phase_deg":
            0.28481322866243236,
            "follower_center_angle_offset_formula":
            "atan((follower_tangential_width_m/2)/pitch_radius_m)",
            "follower_leading_tangent_edge_intersects_start_phase_ray_at_pitch_radius": True,
            "follower_local_center_angles_deg": [
                0.28481322866243236, 120.28481322866243,
                240.28481322866243,
            ],
            "canonical_world_center_angles_deg_after_plug_qX180": [
                -0.28481322866243236, -120.28481322866243,
                -240.28481322866243,
            ],
            "include_segment_start_and_final_shared_endpoint": True,
            "segment_seam_gap_m": 0.0,
            "segment_seam_overlap_m": 0.0,
            "entry_or_end_rounding_allowed": False,
            "rail_solid_direction": "receptacle_local_plus_z",
            "coupling_joint_rotZ_sign_for_positive_coupling": "negative",
            "material_role": "coupling_thread",
            "response_role": "hard_thread",
            "allowed_pair_expansion": "cartesian_all_3_rails_by_all_3_followers",
            "allowed_concrete_pair_count": 9,
            "same_start_index_filtering_allowed": False,
        },
        "thread authoring blueprint",
    )
    electrical = _mapping(blueprint.get("electrical_contacts"), "electrical contacts")
    pins = _mapping(electrical.get("pins"), "pins blueprint")
    _require_exact_fields(
        pins,
        {
            "owner": "FixedReceptacle", "geometry_type": "Cylinder",
            "axis": "local_z", "radius_m": 0.000508,
            "local_z_interval_m": [0.01002, 0.01525],
            "center_local_z_m": 0.012635, "height_m": 0.00523,
            "response_role": "hard_pin",
        },
        "pins blueprint",
    )
    petals = _mapping(electrical.get("socket_petals"), "socket petals blueprint")
    _require_exact_fields(
        petals,
        {
            "owner": "BodyAssembly", "socket_count": 61, "petals_per_socket": 6,
            "expected_total_collider_count": 366,
            "geometry_type": "explicit_convex_tangent_ramp_prism",
            "petal_phase_deg": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
            "tangential_width_m": 0.00018,
            "source_nominal_radial_thickness_before_convex_profile_m": 0.00012,
            "local_depth_breakpoints_m": [0.00200, 0.00220, 0.00280],
            "inner_tangent_distance_at_breakpoints_m": [0.0005080, 0.0004953, 0.0004953],
            "outer_tangent_distance_at_breakpoints_m": [0.0006280, 0.0006153, 0.0006153],
            "convex_depth_radial_profile_points_m": [
                [0.00200, 0.0005080], [0.00220, 0.0004953],
                [0.00280, 0.0004953], [0.00280, 0.0006153],
                [0.00200, 0.0006280],
            ],
            "authored_profile_max_radial_thickness_m": 0.000129525,
            "convex_profile_points_are_authoritative_over_source_nominal_thickness": True,
            "omitted_nonconvex_outer_midpoint_m": [0.00220, 0.0006153],
            "outer_surface_between_front_and_back": "straight_convex_hull_edge",
            "exact_vertex_recipe":
            "ten_vertices_from_five_point_convex_profile_and_two_tangent_planes",
            "one_petal_is_exactly_one_collider": True,
            "response_role": "compliant_socket_petal",
            "nominal_response_k_n_m": 4000.0, "nominal_response_c_n_s_m": 0.10,
            "maximum_physical_deflection_m": 0.000075,
            "first_touch_datum_B_separation_m": 0.01202,
        },
        "socket petals blueprint",
    )
    hard_entries = _mapping(
        electrical.get("socket_hard_entries"), "socket hard entries blueprint"
    )
    if (
        hard_entries.get("expected_instance_count") != 61
        or hard_entries.get("primitive_recipe_id") != "annular_wedge_8v_v1"
        or hard_entries.get("corner_template_id")
        != "axial_band_direct_endpoint_profile_v1"
        or hard_entries.get("expected_convex_piece_count_per_instance") != 72
        or hard_entries.get("expected_total_convex_piece_count") != 4392
        or hard_entries.get("response_role") != "hard_socket_entry"
    ):
        raise ValueError("hard socket-entry inventory changed")
    insert = _mapping(electrical.get("hard_insert_face"), "plug hard insert face")
    if (
        insert.get("outer_polygon_segment_count") != 360
        or insert.get("outer_polygon_phase_deg") != 0.5
        or insert.get("outer_radius_m") != 0.01510
        or insert.get("local_depth_interval_m") != [0.00014, 0.00064]
        or insert.get("hole_count") != 61
        or insert.get("hole_polygon_segment_count") != 24
        or insert.get("required_hole_clear_radius_m") != 0.001105
        or not math.isclose(
            insert.get("hole_polygon_vertex_radius_m"),
            0.001105 / math.cos(math.radians(7.5)), abs_tol=1.0e-15,
        )
        or insert.get("triangulation_rule")
        != "exact_predicate_constrained_Delaunay_without_Steiner_vertices"
        or insert.get("automatic_triangulator_choice_allowed") is not False
        or insert.get("capped_solid_disk_allowed") is not False
    ):
        raise ValueError("true 61-hole plug insert-face blueprint changed")
    maximum_contact_radius = max(
        math.hypot(float(row[1]), float(row[2])) * 0.0254
        for row in EXPECTED_CONTACT_POSITIONS_IN
    )
    if not math.isclose(
        _positive(
            insert.get("minimum_outer_carrier_margin_m"),
            "hard insert outer carrier margin",
        ),
        insert["outer_radius_m"] - maximum_contact_radius - 0.00125,
        abs_tol=1.0e-15,
    ):
        raise ValueError("hard insert outer carrier margin is inconsistent")
    backing = _mapping(electrical.get("fixed_backing_face"), "fixed backing face")
    if (
        backing.get("local_z_interval_m") != [0.01525, 0.01575]
        or backing.get("hole_count") != 61
        or backing.get("required_hole_clear_radius_m") != 0.0011875
        or backing.get("polygon_and_triangulation_rules_identical_to_hard_insert_face")
        is not True
    ):
        raise ValueError("fixed 61-hole backing-face blueprint changed")
    barriers = _mapping(blueprint.get("pin_barriers"), "pin barriers blueprint")
    _require_exact_fields(
        barriers,
        {
            "primitive_recipe_id": "annular_wedge_8v_v1",
            "corner_template_id": "axial_band_direct_endpoint_profile_v1",
            "owner": "FixedReceptacle", "target_owner": "BodyAssembly",
            "expected_instance_count": 61, "angular_wedges_per_instance": 24,
            "axial_profile_bands_per_wedge": 2,
            "expected_total_convex_piece_count": 2928,
            "response_role": "compliant_pin_barrier",
            "nominal_effective_per_barrier_k_n_m": 250.0,
            "nominal_effective_per_barrier_c_n_s_m": 0.020,
            "nominal_authored_per_angular_wedge_k_n_m": 10.416666666666666,
            "nominal_authored_per_angular_wedge_c_n_s_m": 0.0008333333333333334,
            "divisor_is_angular_wedge_count_not_convex_piece_count": True,
            "all_profile_bands_of_one_wedge_share_one_response_material_identity": True,
            "maximum_physical_normal_deflection_m": 0.000335,
            "post_contact_axial_travel_is_not_deflection": True,
            "unreached_profile_cap_is_not_deflection": True,
        },
        "pin barriers blueprint",
    )
    if not math.isclose(
        barriers["nominal_authored_per_angular_wedge_k_n_m"] * 24.0,
        barriers["nominal_effective_per_barrier_k_n_m"], abs_tol=1.0e-12,
    ):
        raise ValueError("pin-barrier blueprint stiffness was not divided by 24")
    spring = _mapping(blueprint.get("spring_fingers"), "spring fingers blueprint")
    _require_exact_fields(
        spring,
        {
            "source_kind": "FROZEN_SIMULATION_PROXY",
            "owner": "BodyAssembly", "target_owner": "FixedReceptacle",
            "finger_count": 12, "finger_phase_formula_deg": "-8-30*segment_index",
            "canonical_world_phase_formula_deg": "8+30*segment_index",
            "target_segment_count": 12,
            "target_phase_formula_deg": "8+30*segment_index",
            "target_tangential_width_m": 0.00160,
            "finger_geometry_type": "explicit_convex_tangent_ramp_prism",
            "finger_tangential_width_m": 0.00160,
            "finger_inner_radius_m": 0.017845,
            "finger_depth_breakpoints_m": [0.01080, 0.01130, 0.01380],
            "finger_outer_radius_at_breakpoints_m": [0.0179575, 0.0180375, 0.0180375],
            "target_inner_bore_radius_m": 0.0179575,
            "target_local_z_interval_m": [0.0, 0.0060],
            "target_piece_shape":
            "one_target_per_finger_explicit_convex_tangent_bore_prism",
            "target_is_collision_isolated_duplicate_proxy": True,
            "all_non_spring_finger_pairs_to_target_are_filtered": True,
            "one_finger_contacts_one_target_at_canonical_yaw": True,
            "onset_datum_B_separation_m": 0.01080,
            "nominal_radial_overlap_m": 0.000080,
            "maximum_physical_radial_deflection_m": 0.00025,
            "keyway_clearance_checked_for_all_12_fingers": True,
            "finger_material_role": "spring_finger",
            "target_material_role": "fixture_and_receptacle",
            "compliant_response_role": "compliant_spring_finger",
            "hard_target_response_role": "hard_receptacle_bore",
            "nominal_response_k_n_m": 12000.0,
            "nominal_response_c_n_s_m": 1.0,
            "only_finger_to_named_bore_target_pairs_enabled": True,
            "nominal_resolved_dynamic_friction_coefficient": 0.25,
            "nominal_friction_axial_force_rationale_n": 2.88,
            "nominal_friction_axial_force_rationale_formula":
            "12*12000*0.000080*max(0.20,0.25)",
            "rationale_is_not_A3_pass_value": True,
        },
        "spring fingers blueprint",
    )
    seal = _mapping(blueprint.get("peripheral_seal"), "peripheral seal blueprint")
    _require_exact_fields(
        seal,
        {
            "primitive_recipe_id": "annular_wedge_8v_v1",
            "corner_template_id": "constant_direct_endpoint_interval_v1",
            "source_kind": "FROZEN_SIMULATION_PROXY",
            "seal_owner": "FixedReceptacle", "target_owner": "BodyAssembly",
            "segment_count": 24, "angular_step_deg": 15.0,
            "phase_origin_deg": 7.5, "radial_interval_m": [0.01515, 0.01635],
            "seal_local_z_interval_m": [0.014615, 0.015615],
            "seal_contact_face_bound": "minZ",
            "target_local_depth_interval_m": [0.0, 0.0010],
            "target_contact_face_bound": "minZ",
            "material_role_compliant_side": "peripheral_seal",
            "material_role_hard_side": "plug_shell_and_keys",
            "compliant_response_role": "compliant_peripheral_seal",
            "hard_target_response_role": "hard_seal_target",
            "nominal_response_k_n_m": 800.0, "nominal_response_c_n_s_m": 0.5,
            "first_touch_datum_B_separation_m": 0.014615,
            "nominal_deflection_at_bottoming_m": 0.000435,
            "only_named_seal_to_target_pairs_enabled": True,
        },
        "peripheral seal blueprint",
    )
    detent = _mapping(
        blueprint.get("anti_decoupling_detent"), "detent authoring blueprint"
    )
    _require_exact_fields(
        detent,
        {
            "representation":
            "continuous_analytic_base_cylinder_plus_36_convex_tooth_prisms_and_three_compliant_followers",
            "rejected_r8_representation":
            "1368_separately_closed_annular_wedges_with_false_internal_partition_faces",
            "rejected_r8_representation_may_not_be_reused": True,
            "cam_owner": "BodyAssembly", "follower_owner": "CouplingNut",
            "cam_base_path":
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/AntiDecoupling/Cam/ContinuousBase",
            "cam_base_geometry_type": "analytic_cylinder",
            "cam_base_primitive_recipe_id": "analytic_cylinder_v1",
            "cam_tooth_path_template":
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/AntiDecoupling/Cam/Teeth/Tooth_{{tooth_index:02d}}",
            "cam_tooth_primitive_recipe_id": "planar_triangle_z_prism_6v_v1",
            "tooth_count": 36, "pitch_per_tooth_deg": 10.0,
            "dwell_span_deg": 8.982274,
            "positive_coupling_ascent_span_deg": 0.926547,
            "reverse_drop_span_deg": 0.091179,
            "expected_cam_base_count": 1,
            "expected_cam_tooth_count": 36,
            "expected_cam_total_collider_count": 37,
            "dwell_contact_is_continuous_analytic_cylinder": True,
            "separately_closed_dwell_wedges_allowed": False,
            "exposed_internal_partition_faces_allowed": False,
            "tooth_solids_intentionally_overlap_same_owner_continuous_base": True,
            "tooth_base_chord_is_buried_inside_continuous_base": True,
            "tooth_exposed_contact_edges_are_straight_chords_between_declared_polar_vertices": True,
            "analytic_hardware_sawtooth_curve_claimed": False,
            "shared_local_center_z_m": 0.0200,
            "cam_local_center_z_m": 0.0200,
            "follower_local_center_z_m": 0.0200,
            "cam_axial_width_m": 0.00080,
            "cam_outer_base_radius_m": 0.021975,
            "cam_outer_peak_radius_m": 0.022025,
            "cam_tooth_local_z_interval_m": [0.01960, 0.02040],
            "cam_tooth_CCW_vertex_order": [
                {
                    "vertex": 0,
                    "positive_coupling_progress_deg": 10.0,
                    "local_theta_formula_deg": "tooth_phase_deg-10.0",
                    "radius_m": 0.021975,
                },
                {
                    "vertex": 1,
                    "positive_coupling_progress_deg": 9.908821,
                    "local_theta_formula_deg": "tooth_phase_deg-9.908821",
                    "radius_m": 0.022025,
                },
                {
                    "vertex": 2,
                    "positive_coupling_progress_deg": 8.982274,
                    "local_theta_formula_deg": "tooth_phase_deg-8.982274",
                    "radius_m": 0.021975,
                },
            ],
            "cam_tooth_point_order": [
                "vertex0_zLow", "vertex1_zLow", "vertex2_zLow",
                "vertex0_zHigh", "vertex1_zHigh", "vertex2_zHigh",
            ],
            "cam_tooth_faceVertexCounts": [3, 3, 4, 4, 4],
            "cam_tooth_faceVertexIndices": [
                2, 1, 0, 3, 4, 5, 0, 1, 4, 3,
                1, 2, 5, 4, 2, 0, 3, 5,
            ],
            "cam_tooth_positive_volume_required": True,
            "follower_count": 3,
            "follower_phases_deg": [-4.491137, 115.508863, 235.508863],
            "follower_phase_offset_from_tooth_origin_deg": -4.491137,
            "follower_shape": "radial_thin_tangent_box_convex_prism",
            "follower_primitive_recipe_id": "radial_tangent_box_8v_v1",
            "follower_contact_surface":
            "inner_radial_tangent_face_of_0p020mm_wide_box",
            "follower_axial_width_m": 0.00060,
            "follower_tangential_width_m": 0.000020,
            "follower_radial_interval_m": [0.021974, 0.022749],
            "base_overlap_m": 0.000001,
            "peak_overlap_m": 0.000051,
            "cam_response_role": "hard_detent_cam",
            "follower_response_role": "compliant_detent_follower",
            "friction_coefficients_must_be_zero": True,
            "positive_coupling_joint_rotZ_sign": "negative",
            "positive_coupling_progress_formula_deg":
            "p=wrap(-theta_body_local_deg+tooth0_phase_origin_deg,0,10)",
            "profile_in_positive_coupling_progress": [
                {
                    "name": "base_dwell", "p_start_deg": 0.0,
                    "p_end_deg": 8.982274,
                    "outer_radius_start_m": 0.021975,
                    "outer_radius_end_m": 0.021975,
                },
                {
                    "name": "shallow_ascent", "p_start_deg": 8.982274,
                    "p_end_deg": 9.908821,
                    "outer_radius_start_m": 0.021975,
                    "outer_radius_end_m": 0.022025,
                },
                {
                    "name": "steep_reverse_drop", "p_start_deg": 9.908821,
                    "p_end_deg": 10.0,
                    "outer_radius_start_m": 0.022025,
                    "outer_radius_end_m": 0.021975,
                },
            ],
            "followers_start_at_center_of_continuous_base_dwell": True,
            "nominal_follower_response_k_n_m": 110000.0,
            "nominal_follower_response_c_n_s_m": 2.0,
            "only_continuous_cam_base_and_teeth_to_three_follower_pairs_enabled": True,
        },
        "detent authoring blueprint",
    )
    if not math.isclose(
        detent["dwell_span_deg"] + detent["positive_coupling_ascent_span_deg"]
        + detent["reverse_drop_span_deg"], 10.0, abs_tol=1.0e-6,
    ):
        raise ValueError("detent tooth angular pieces do not close one pitch")
    bottom = _mapping(blueprint.get("metal_bottoming"), "metal bottoming blueprint")
    _require_exact_fields(
        bottom,
        {
            "primitive_recipe_id": "annular_wedge_8v_v1",
            "corner_template_id": "constant_direct_endpoint_interval_v1",
            "source_kind": "FROZEN_SIMULATION_PROXY",
            "fixed_owner": "FixedReceptacle", "plug_owner": "BodyAssembly",
            "segment_count": 48, "angular_step_deg": 7.5,
            "phase_origin_deg": 3.75, "radial_interval_m": [0.01655, 0.01695],
            "axial_thickness_m": 0.00030,
            "fixed_local_z_interval_m": [0.0, 0.00030],
            "fixed_contact_face_bound": "minZ",
            "plug_nominal_local_depth_interval_m": [0.01505, 0.01535],
            "plug_contact_face_bound": "minZ",
            "variant_plug_contact_depth_m": {
                "minimum_interference": 0.01501,
                "nominal": 0.01505,
                "maximum_interference": 0.01509,
            },
            "response_role": "hard_metal_bottoming",
            "fixed_material_role": "fixture_and_receptacle",
            "plug_material_role": "plug_shell_and_keys",
            "only_named_stop_group_to_stop_group_pairs_enabled": True,
        },
        "metal bottoming blueprint",
    )
    shoulders = _mapping(blueprint.get("nut_body_shoulders"), "nut/body shoulders")
    if (
        shoulders.get("group_count") != 4
        or shoulders.get("primitive_recipe_id") != "annular_wedge_8v_v1"
        or shoulders.get("corner_template_id")
        != "constant_direct_endpoint_interval_v1"
        or shoulders.get("wedge_count_per_group") != 48
        or shoulders.get("expected_total_convex_piece_count") != 192
        or shoulders.get("response_role") != "hard_nut_body_shoulder"
        or shoulders.get("exact_enabled_pairs")
        != ["positive_body0_to_positive_body1", "negative_body0_to_negative_body1"]
        or shoulders.get("exact_filtered_cross_pairs")
        != ["positive_body0_to_negative_body1", "negative_body0_to_positive_body1"]
    ):
        raise ValueError("nut/body shoulder piece or pair blueprint changed")
    filtering = _mapping(blueprint.get("filtering"), "connector filtering")
    if (
        filtering.get("authoring_scope")
        != "leaf_collider_pairs_or_family_collision_groups_only"
        or filtering.get("FilteredPairsAPI_on_BodyAssembly_or_CouplingNut_root_allowed")
        is not False
        or filtering.get("ancestor_filter_that_covers_an_intended_pair_allowed")
        is not False
        or filtering.get("same_label_contact_filtering_allowed") is not False
        or filtering.get("wrong_key_yaw_filtering_allowed") is not False
        or filtering.get("default_unlisted_connector_pair_decision") != "filtered"
        or filtering.get("every_concrete_pair_matches_exactly_one_final_rule") is not True
        or filtering.get("unmatched_concrete_pair_count_allowed") != 0
        or filtering.get("multiply_matched_concrete_pair_count_allowed") != 0
        or filtering.get("joint_collision_gate") != {
            "joint": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNutJoint",
            "physics_collisionEnabled": True,
        }
    ):
        raise ValueError("leaf-only connector collision filtering contract changed")

    partition = _mapping(
        filtering.get("primitive_family_partition"),
        "connector primitive-family partition",
    )
    if partition != {
        "every_connector_leaf_belongs_to_exactly_one_primitive_family": True,
        "primitive_family_count": 28,
        "expected_total_leaf_count_nominal": 15037,
        "expected_total_unordered_distinct_leaf_pair_count_nominal": 113048166,
        "expected_unordered_primitive_family_pair_count": 406,
        "path_match_is_full_match_not_prefix_guess": True,
        "owner_role_and_response_must_equal_family_definition": True,
        "runtime_generated_or_unclassified_connector_collider_allowed": False,
    }:
        raise ValueError("connector primitive-family partition contract changed")
    primitive_families = _mapping(
        filtering.get("primitive_family_definitions"),
        "connector primitive-family definitions",
    )
    if len(primitive_families) != partition["primitive_family_count"]:
        raise ValueError("connector primitive-family count changed")
    leaf_counts: dict[str, int] = {}
    all_path_templates: set[str] = set()
    for family_name, raw_family in primitive_families.items():
        family = _mapping(raw_family, f"primitive family {family_name}")
        count = family.get("expected_leaf_count_nominal")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"primitive family {family_name} has invalid leaf count")
        if family.get("owner") not in {"FixedReceptacle", "BodyAssembly", "CouplingNut"}:
            raise ValueError(f"primitive family {family_name} has invalid rigid owner")
        if not isinstance(family.get("material_role"), str) or not family["material_role"]:
            raise ValueError(f"primitive family {family_name} lacks one material role")
        if not isinstance(family.get("response_role"), str) or not family["response_role"]:
            raise ValueError(f"primitive family {family_name} lacks one response role")
        paths = family.get("path_templates")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"primitive family {family_name} lacks path templates")
        for path_template in paths:
            if (
                not isinstance(path_template, str)
                or not path_template.startswith(f"{SUCCESSOR_ROOT_PRIM}/")
                or path_template in all_path_templates
            ):
                raise ValueError(
                    f"primitive family {family_name} has invalid or duplicate path template"
                )
            all_path_templates.add(path_template)
        if not isinstance(family.get("index_domains"), dict):
            raise ValueError(f"primitive family {family_name} lacks exact index domains")
        leaf_counts[family_name] = count
    total_leaf_count = sum(leaf_counts.values())
    if total_leaf_count != partition["expected_total_leaf_count_nominal"]:
        raise ValueError("connector primitive-family leaf counts do not close")
    if total_leaf_count * (total_leaf_count - 1) // 2 != partition[
        "expected_total_unordered_distinct_leaf_pair_count_nominal"
    ]:
        raise ValueError("connector total unordered leaf-pair count does not close")
    if len(leaf_counts) * (len(leaf_counts) + 1) // 2 != partition[
        "expected_unordered_primitive_family_pair_count"
    ]:
        raise ValueError("connector unordered primitive-family pair count does not close")

    leaf_readback = _mapping(
        filtering.get("realized_leaf_readback_contract"),
        "connector realized-leaf readback contract",
    )
    if leaf_readback != {
        "collider_index_per_leaf_prim": 0,
        "collisionEnabled": True,
        "required_collision_schema": "PhysicsCollisionAPI",
        "default_typeName": "Mesh",
        "default_geometry_type": "explicit_convex_mesh",
        "default_physics_approximation": "convexHull",
        "analytic_primitive_overrides": {
            "pins_61": {
                "typeName": "Cylinder",
                "geometry_type": "analytic_cylinder",
                "physics_approximation": "none",
            },
            "detent_cam_continuous_base_1": {
                "typeName": "Cylinder",
                "geometry_type": "analytic_cylinder",
                "physics_approximation": "none",
            },
        },
        "closed_manifold": True,
        "positive_volume": True,
        "convex": True,
        "owner_rigidBodyEnabled": True,
        "owner_kinematicEnabled": False,
        "offset_class": "fine_connector",
        "contactOffset_m": 0.00001,
        "restOffset_m": 0.0,
        "exactly_one_collision_group_membership_per_leaf": True,
        "collision_group_path_template": f"{SUCCESSOR_ROOT_PRIM}/CollisionGroups/{{primitive_family}}",
        "topology_signature_is_structured_semantic_text_not_a_fingerprint": True,
        "empty_or_fingerprint_topology_signature_allowed": False,
    }:
        raise ValueError("connector realized-leaf readback contract changed")
    group_authoring = _mapping(
        filtering.get("collision_group_authoring"),
        "connector collision-group authoring contract",
    )
    if group_authoring != {
        "group_root": f"{SUCCESSOR_ROOT_PRIM}/CollisionGroups",
        "group_path_template": f"{SUCCESSOR_ROOT_PRIM}/CollisionGroups/{{primitive_family}}",
        "expected_group_count": 28,
        "group_schema": "PhysicsCollisionGroup",
        "member_collection_relationship": "collection:colliders:includes",
        "filtered_group_relationship": "physics:filteredGroups",
        "exactly_one_group_membership_per_leaf": True,
        "merge_group_name_authored_allowed": False,
        "filtered_base_family_pair_count": 387,
        "enabled_base_family_pair_count": 19,
        "expected_filter_source_row_count": 387,
        "unordered_pair_source_group": "lexicographically_smaller_family_name",
        "unordered_pair_target_group": "lexicographically_larger_family_name",
        "same_family_filter_uses_self_target": True,
        "duplicate_symmetric_filter_relationship_allowed": False,
        "ancestor_or_leaf_FilteredPairsAPI_allowed": False,
        "enabled_pair_must_have_no_filter_source": True,
    }:
        raise ValueError("connector collision-group authoring contract changed")

    composite_families = _mapping(
        filtering.get("composite_family_definitions"),
        "connector composite-family definitions",
    )
    primitive_names = set(leaf_counts)

    def _resolve_family(name: str) -> set[str]:
        if name in primitive_names:
            return {name}
        if name not in composite_families:
            raise ValueError(f"unknown connector collider family {name}")
        definition = _mapping(
            composite_families[name], f"composite collider family {name}"
        )
        include_all = definition.get("include_all_primitive_families", False)
        if include_all not in {True, False}:
            raise ValueError(f"composite collider family {name} has invalid include-all flag")
        included = set(primitive_names) if include_all else set(definition.get("include", []))
        excluded = set(definition.get("exclude", []))
        if not included <= primitive_names or not excluded <= primitive_names:
            raise ValueError(f"composite collider family {name} references an unknown family")
        resolved = included - excluded
        if not resolved:
            raise ValueError(f"composite collider family {name} resolves empty")
        return resolved

    for composite_name in composite_families:
        _resolve_family(composite_name)

    expansion_contract = _mapping(
        filtering.get("rule_expansion_contract"),
        "connector family-pair expansion contract",
    )
    if expansion_contract.get("cartesian") != (
        "all_distinct_leaf_cross_pairs_canonicalized_as_unordered_pairs"
    ) or expansion_contract.get("same_primitive_family_pair_cardinality") != (
        "n_times_n_minus_1_over_2"
    ) or expansion_contract.get("different_primitive_family_pair_cardinality") != (
        "n_left_times_n_right"
    ) or expansion_contract.get("default_rule_expansion") != (
        "unordered_complement_of_all_prior_explicit_rules"
    ):
        raise ValueError("connector family-pair expansion semantics changed")
    declared_cross_pairs = _mapping(
        expansion_contract.get("declared_cross_pairs"),
        "declared connector family cross-pairs",
    )
    if declared_cross_pairs != {
        "SHOULDER_CROSS_FILTER": [
            ["shoulder_positive_body0_48", "shoulder_negative_body1_48"],
            ["shoulder_negative_body0_48", "shoulder_positive_body1_48"],
        ]
    }:
        raise ValueError("declared connector family cross-pairs changed")

    expected_rules = {
        "THREAD_ALL_STARTS", "PIN_TO_HARD_ENTRY_ALL", "PIN_TO_PETALS_ALL",
        "PIN_TO_INSERT_FACE", "BARRIER_TO_ENTRY_ALL", "SEAL_TO_TARGET_ONLY",
        "SPRING_TO_BORE_ONLY", "DETENT_ONLY", "KEYS_AND_SHELLS_GEOMETRIC",
        "FIXED_BACKING_TO_PLUG_FACE", "NUT_ENVELOPE_WRONG_POSE_BLOCK",
        "BOTTOMING_ONLY", "SHOULDER_POSITIVE", "SHOULDER_NEGATIVE",
        "SHOULDER_CROSS_FILTER", "BODY_NUT_OTHER_FILTER",
        "BORE_TARGET_ALL_OTHER_FILTER", "REAR_BODY_FRONT_NONSTOP_FILTER",
        "DEFAULT_UNLISTED_FILTER",
    }
    rules = filtering.get("family_pair_rules")
    if not isinstance(rules, list):
        raise ValueError("family-pair filtering rules must be a list")
    by_id = {
        _mapping(rule, "family-pair rule").get("rule_id"): rule for rule in rules
    }
    if set(by_id) != expected_rules or len(by_id) != len(rules):
        raise ValueError("family-pair filtering rule inventory changed")
    expected_rule_values = {
        "THREAD_ALL_STARTS": ("thread_rails_3", "thread_followers_3", "cartesian", "enabled", "hard_frictional"),
        "PIN_TO_HARD_ENTRY_ALL": ("pins_61", "hard_socket_entries_61", "cartesian", "enabled", "hard"),
        "PIN_TO_PETALS_ALL": ("pins_61", "socket_petals_366", "cartesian", "enabled", "compliant"),
        "PIN_TO_INSERT_FACE": ("pins_61", "hard_insert_face_prisms", "cartesian", "enabled", "hard"),
        "BARRIER_TO_ENTRY_ALL": ("pin_barriers_61", "hard_socket_entries_61", "cartesian", "enabled", "compliant"),
        "SEAL_TO_TARGET_ONLY": ("seal_segments_24", "seal_targets_24", "cartesian", "enabled", "compliant"),
        "SPRING_TO_BORE_ONLY": ("spring_fingers_12", "receptacle_bore_targets_12", "cartesian", "enabled", "compliant"),
        "BORE_TARGET_ALL_OTHER_FILTER": ("receptacle_bore_targets_12", "all_connector_colliders_except_spring_fingers_12", "cartesian", "filtered", "none"),
        "DETENT_ONLY": ("detent_cam_contact_surfaces", "detent_followers_3", "cartesian", "enabled", "compliant"),
        "KEYS_AND_SHELLS_GEOMETRIC": ("plug_keys_and_shell", "receptacle_keyway_walls_and_shell", "cartesian", "enabled", "hard"),
        "FIXED_BACKING_TO_PLUG_FACE": ("fixed_backing_face_prisms", "hard_insert_face_prisms", "cartesian", "enabled", "hard"),
        "NUT_ENVELOPE_WRONG_POSE_BLOCK": ("coupling_nut_envelope", "receptacle_shell_and_thread_carrier", "cartesian", "enabled", "hard"),
        "REAR_BODY_FRONT_NONSTOP_FILTER": ("body_assembly_rear_body_288", "receptacle_keyway_shell_only", "cartesian", "filtered", "none"),
        "BOTTOMING_ONLY": ("fixed_metal_stop_48", "plug_metal_stop_48", "cartesian", "enabled", "hard"),
        "SHOULDER_POSITIVE": ("shoulder_positive_body0_48", "shoulder_positive_body1_48", "cartesian", "enabled", "hard"),
        "SHOULDER_NEGATIVE": ("shoulder_negative_body0_48", "shoulder_negative_body1_48", "cartesian", "enabled", "hard"),
        "SHOULDER_CROSS_FILTER": ("shoulder_opposite_sign_groups", "shoulder_opposite_sign_groups", "declared_cross_pairs", "filtered", "none"),
        "BODY_NUT_OTHER_FILTER": ("other_BodyAssembly_leaf_colliders", "other_CouplingNut_leaf_colliders", "cartesian", "filtered", "none"),
        "DEFAULT_UNLISTED_FILTER": ("all_connector_colliders", "all_connector_colliders", "unordered_complement_of_all_prior_explicit_rules", "filtered", "none"),
    }
    for rule_id, (left, right, expansion, decision, response_name) in expected_rule_values.items():
        if dict(by_id[rule_id]) != {
            "rule_id": rule_id,
            "left": left,
            "right": right,
            "expansion": expansion,
            "final_decision": decision,
            "response": response_name,
        }:
            raise ValueError(f"collider family-pair rule {rule_id} changed")
    for rule_id in {
        "THREAD_ALL_STARTS", "PIN_TO_HARD_ENTRY_ALL", "PIN_TO_PETALS_ALL",
        "PIN_TO_INSERT_FACE", "BARRIER_TO_ENTRY_ALL", "SEAL_TO_TARGET_ONLY",
        "SPRING_TO_BORE_ONLY", "DETENT_ONLY", "KEYS_AND_SHELLS_GEOMETRIC",
        "FIXED_BACKING_TO_PLUG_FACE", "NUT_ENVELOPE_WRONG_POSE_BLOCK",
        "BOTTOMING_ONLY", "SHOULDER_POSITIVE", "SHOULDER_NEGATIVE",
    }:
        if by_id[rule_id].get("final_decision") != "enabled":
            raise ValueError(f"intended collider family pair {rule_id} is filtered")
    for rule_id in {
        "SHOULDER_CROSS_FILTER", "BODY_NUT_OTHER_FILTER",
        "BORE_TARGET_ALL_OTHER_FILTER", "REAR_BODY_FRONT_NONSTOP_FILTER",
        "DEFAULT_UNLISTED_FILTER",
    }:
        if by_id[rule_id].get("final_decision") != "filtered":
            raise ValueError(f"forbidden internal collider pair {rule_id} is enabled")

    def _canonical_family_pair(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left <= right else (right, left)

    all_base_pairs = {
        _canonical_family_pair(left, right)
        for index, left in enumerate(sorted(primitive_names))
        for right in sorted(primitive_names)[index:]
    }
    explicit_rule_pairs: dict[str, set[tuple[str, str]]] = {}
    already_matched: set[tuple[str, str]] = set()
    for rule in rules:
        rule_id = rule["rule_id"]
        expansion = rule["expansion"]
        if expansion == "unordered_complement_of_all_prior_explicit_rules":
            continue
        if expansion == "cartesian":
            pairs = {
                _canonical_family_pair(left, right)
                for left in _resolve_family(rule["left"])
                for right in _resolve_family(rule["right"])
            }
        elif expansion == "declared_cross_pairs":
            pairs = {
                _canonical_family_pair(left, right)
                for left, right in declared_cross_pairs[rule_id]
            }
        else:
            raise ValueError(f"unsupported connector family expansion {expansion}")
        if not pairs or not pairs <= all_base_pairs:
            raise ValueError(f"connector rule {rule_id} has invalid base-family pairs")
        overlap = already_matched & pairs
        if overlap:
            raise ValueError(
                f"connector family rules multiply match {len(overlap)} base-family pairs"
            )
        explicit_rule_pairs[rule_id] = pairs
        already_matched |= pairs
    default_pairs = all_base_pairs - already_matched
    explicit_rule_pairs["DEFAULT_UNLISTED_FILTER"] = default_pairs

    def _leaf_pair_cardinality(pair: tuple[str, str]) -> int:
        left, right = pair
        if left == right:
            count = leaf_counts[left]
            return count * (count - 1) // 2
        return leaf_counts[left] * leaf_counts[right]

    explicit_base_count = len(already_matched)
    default_base_count = len(default_pairs)
    explicit_leaf_pair_count = sum(
        _leaf_pair_cardinality(pair) for pair in already_matched
    )
    default_leaf_pair_count = sum(
        _leaf_pair_cardinality(pair) for pair in default_pairs
    )
    expected_expansion_counts = {
        "explicit_rule_base_family_pair_sets_must_be_disjoint": True,
        "union_with_default_must_equal_all_406_unordered_base_family_pairs": True,
        "expected_explicit_base_family_pair_count": 76,
        "expected_default_base_family_pair_count": 330,
        "expected_explicit_concrete_leaf_pair_count": 19366897,
        "expected_default_concrete_leaf_pair_count": 93681269,
    }
    for field, expected_value in expected_expansion_counts.items():
        if expansion_contract.get(field) != expected_value:
            raise ValueError(f"connector family expansion field {field} changed")
    if (
        explicit_base_count != expansion_contract["expected_explicit_base_family_pair_count"]
        or default_base_count != expansion_contract["expected_default_base_family_pair_count"]
        or explicit_leaf_pair_count
        != expansion_contract["expected_explicit_concrete_leaf_pair_count"]
        or default_leaf_pair_count
        != expansion_contract["expected_default_concrete_leaf_pair_count"]
        or explicit_base_count + default_base_count != len(all_base_pairs)
        or explicit_leaf_pair_count + default_leaf_pair_count
        != partition["expected_total_unordered_distinct_leaf_pair_count_nominal"]
    ):
        raise ValueError("connector family-pair symbolic cardinalities do not close")
    enabled_base_pair_count = sum(
        len(explicit_rule_pairs[rule_id])
        for rule_id, rule in by_id.items()
        if rule["final_decision"] == "enabled"
    )
    filtered_base_pair_count = len(all_base_pairs) - enabled_base_pair_count
    if (
        enabled_base_pair_count != group_authoring["enabled_base_family_pair_count"]
        or filtered_base_pair_count
        != group_authoring["filtered_base_family_pair_count"]
        or filtered_base_pair_count
        != group_authoring["expected_filter_source_row_count"]
    ):
        raise ValueError("collision-group enabled/filtered family counts do not close")
    if blueprint != FROZEN_A2_COLLISION_AUTHORING_BLUEPRINT:
        raise ValueError(
            "complete A2 collision-authoring blueprint differs from its "
            "independent frozen literal snapshot"
        )


def _validate_materials_and_solver(document: Mapping[str, Any]) -> None:
    materials = _mapping(document.get("material_roles"), "material_roles")
    if materials.get("binding_policy") != "explicit_role_metadata_only_no_path_blanket_binding":
        raise ValueError("materials must be bound by explicit role metadata")
    if (
        materials.get("role_metadata_attribute") != "kcg:materialRole"
        or materials.get("response_metadata_attribute") != "kcg:responseRole"
        or materials.get("every_collision_prim_requires_exactly_one_role") is not True
        or materials.get("every_collision_prim_requires_exactly_one_response_role")
        is not True
        or materials.get("unassigned_collision_prim_count_allowed") != 0
        or materials.get("unassigned_response_role_collider_count_allowed") != 0
        or materials.get("path_or_name_blanket_rebinding_allowed") is not False
    ):
        raise ValueError("material-role metadata or fail-closed binding policy changed")
    if materials.get("combine_modes") != {
        "friction": "max",
        "restitution": "min",
        "compliant_damping": "max",
    }:
        raise ValueError("semantic material combine modes changed")
    roles = _mapping(materials.get("roles"), "material_roles.roles")
    if dict(roles) != EXPECTED_MATERIAL_ROLE_VALUES:
        raise ValueError("material friction/restitution role values changed")
    for name, raw in roles.items():
        values = _mapping(raw, f"material role {name}")
        static = _finite(values.get("static_friction"), f"{name}.static_friction")
        dynamic = _finite(values.get("dynamic_friction"), f"{name}.dynamic_friction")
        restitution = _finite(values.get("restitution"), f"{name}.restitution")
        if dynamic < 0.0 or static < dynamic or not 0.0 <= restitution <= 1.0:
            raise ValueError(f"material role {name} is not physically admissible")
    if materials.get("semantic_friction_and_restitution_values_immutable_through_A2_A3") is not True:
        raise ValueError("hard and frictional material values must remain frozen through A3")
    response_roles = _mapping(
        materials.get("response_roles"), "material response roles"
    )
    if set(response_roles) != (
        set(REQUIRED_HARD_RESPONSE_ROLES) | set(REQUIRED_COMPLIANT_RESPONSE_VALUES)
    ):
        raise ValueError("hard/compliant response-role inventory changed")
    for name in REQUIRED_HARD_RESPONSE_ROLES:
        if response_roles.get(name) != {
            "class": "hard",
            "compliant_stiffness_n_m": 0.0,
            "compliant_damping_n_s_m": 0.0,
        }:
            raise ValueError(f"hard response role {name} became compliant")
    for name, (stiffness, damping) in REQUIRED_COMPLIANT_RESPONSE_VALUES.items():
        role = _mapping(response_roles.get(name), f"response role {name}")
        if (
            role.get("class") != "compliant"
            or _positive(role.get("nominal_stiffness_n_m"), f"{name} stiffness")
            != stiffness
            or _positive(role.get("nominal_damping_n_s_m"), f"{name} damping")
            != damping
            or role.get("accelerationSpring") is not False
        ):
            raise ValueError(f"compliant response role {name} changed")
    if response_roles["compliant_pin_barrier"].get("parameter_scope") != "per_angular_wedge":
        raise ValueError("pin-barrier response material is not per angular wedge")
    if materials.get("compliant_pair_policy") != {
        "exactly_one_compliant_side": True,
        "hard_side_stiffness_and_damping_are_zero": True,
        "resolved_effective_stiffness_and_damping_required": True,
    }:
        raise ValueError("exactly-one-compliant-side material policy changed")
    link_roles = _mapping(
        materials.get("robot_collision_link_roles"), "robot collision-link roles"
    )
    expected_link_roles = {
        "robot_structure": [
            "iiwa_link_0", "iiwa_link_1", "iiwa_link_2", "iiwa_link_3",
            "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7",
            "handbase_link",
        ],
        "finger_structure": [
            "f1Link1", "f1Link2", "f2Link1", "f3Link1", "f3Link2",
        ],
        "fingertip_pad": ["f1Link3", "f2Link2", "f3Link3"],
    }
    if dict(link_roles) != expected_link_roles:
        raise ValueError("robot/finger/fingertip material-role partition changed")
    assigned_links = [link for values in link_roles.values() for link in values]
    if len(assigned_links) != 17 or len(set(assigned_links)) != 17:
        raise ValueError("17 robot collision links must have one unique material role")
    fingertip = _mapping(
        materials.get("fingertip_geometry_policy"), "fingertip geometry policy"
    )
    if (
        fingertip.get("representation")
        != "entire_existing_terminal_collision_hull_as_final_grip_surface_proxy"
        or fingertip.get("separate_pad_geometry_claimed") is not False
        or fingertip.get("terminal_links") != ["f1Link3", "f2Link2", "f3Link3"]
        or fingertip.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("terminal grip-surface proxy changed")
    solver = _mapping(document.get("solver_profile"), "solver_profile")
    if solver.get("source_kind") != "FROZEN_SIMULATION_PROXY":
        raise ValueError("solver settings must be explicit simulation proxies")
    if solver.get("schema_versions") != {
        "omni_usd_schema_physx": "110.1.13",
        "omni_usd_schema_newton": "1.2.1",
    }:
        raise ValueError("local Isaac physics schema versions changed")
    if solver.get("physics_rate_hz") != 240:
        raise ValueError("physics rate must remain 240 Hz")
    if not math.isclose(
        _positive(solver.get("physics_dt_s"), "physics_dt_s"), 1.0 / 240.0,
        rel_tol=0.0, abs_tol=1.0e-15,
    ):
        raise ValueError("physics dt differs from 1/240 second")
    if solver.get("solver_type") != "TGS":
        raise ValueError("fine connector profile must use TGS")
    if solver.get("position_iterations") < 16 or solver.get("velocity_iterations") < 4:
        raise ValueError("solver iterations are below the frozen lower bound")
    if solver.get("ccd_required_for_dynamic_connector_and_fingertips") is not True:
        raise ValueError("fine dynamic connector contacts require CCD")
    if _finite(solver.get("rest_offset_m"), "rest_offset_m") != 0.0:
        raise ValueError("rest offset must remain zero")
    authored = _mapping(
        solver.get("authored_attribute_contract"),
        "solver_profile.authored_attribute_contract",
    )
    if authored.get("stage_metadata") != {
        "metersPerUnit": 1.0,
        "kilogramsPerUnit": 1.0,
    }:
        raise ValueError("stage unit metadata changed")
    expected_scene = {
        "physxScene:solverType": "TGS",
        "physxScene:enableCCD": True,
        "physxScene:enableStabilization": True,
        "physxScene:enableEnhancedDeterminism": True,
        "physxScene:enableExternalForcesEveryIteration": True,
        "physxScene:minPositionIterationCount": 32,
        "physxScene:minVelocityIterationCount": 8,
        "physxScene:timeStepsPerSecond": 240,
    }
    if authored.get("physics_scene") != expected_scene:
        raise ValueError("physics-scene authored attribute contract changed")
    expected_rigid = {
        "physxRigidBody:solverPositionIterationCount": 32,
        "physxRigidBody:solverVelocityIterationCount": 8,
        "physxRigidBody:enableCCD": True,
        "physxRigidBody:maxDepenetrationVelocity": 0.20,
    }
    if authored.get("dynamic_rigid_bodies") != expected_rigid:
        raise ValueError("dynamic rigid-body authored attribute contract changed")
    if authored.get("fine_connector_colliders") != {
        "physxCollision:contactOffset": 0.00001,
        "physxCollision:restOffset": 0.0,
    }:
        raise ValueError("fine connector collision offsets changed")
    if authored.get("general_colliders") != {
        "physxCollision:contactOffset": 0.00005,
        "physxCollision:restOffset": 0.0,
    }:
        raise ValueError("general collision offsets changed")
    if authored.get("materials") != {
        "physxMaterial:frictionCombineMode": "max",
        "physxMaterial:restitutionCombineMode": "min",
        "physxMaterial:dampingCombineMode": "max",
    }:
        raise ValueError("material combine-mode attributes changed")
    compliant = _mapping(
        authored.get("compliant_materials"), "compliant material API contract"
    )
    if compliant != {
        "prim_type": "Material",
        "required_applied_schemas": ["PhysicsMaterialAPI", "PhysxMaterialAPI"],
        "physxMaterial:compliantContactStiffness": "positive_role_value_n_m",
        "physxMaterial:compliantContactDamping": "nonnegative_role_value_n_s_m",
        "physxMaterial:compliantContactAccelerationSpring": False,
        "authoring_api": "PhysxSchema.PhysxMaterialAPI",
        "experimental_set_enabled_compliant_contacts_allowed": False,
    }:
        raise ValueError("compliant-contact material API contract changed")
    d6 = _mapping(authored.get("nut_body_D6_joint"), "nut/body D6 contract")
    if d6 != {
        "typeName": "PhysicsJoint",
        "physics:jointEnabled": True,
        "physics:collisionEnabled": True,
        "physics:body0": (
            "/World/D38999Shell25JKeyedPhysicalV3/LoosePlug/BodyAssembly"
        ),
        "physics:body1": (
            "/World/D38999Shell25JKeyedPhysicalV3/LoosePlug/CouplingNut"
        ),
        "physics:localPos0": [0.0, 0.0, 0.020],
        "physics:localPos1": [0.0, 0.0, 0.020],
        "physics:localRot0_wxyz": [1.0, 0.0, 0.0, 0.0],
        "physics:localRot1_wxyz": [1.0, 0.0, 0.0, 0.0],
        "required_applied_schemas": [
            "PhysicsLimitAPI:transX",
            "PhysicsLimitAPI:transY",
            "PhysicsLimitAPI:transZ",
            "PhysicsLimitAPI:rotX",
            "PhysicsLimitAPI:rotY",
        ],
        "limit:transX:physics:low": 1.0,
        "limit:transX:physics:high": -1.0,
        "limit:transY:physics:low": 1.0,
        "limit:transY:physics:high": -1.0,
        "limit:transZ:physics:low": -0.00015,
        "limit:transZ:physics:high": 0.00015,
        "limit:rotX:physics:low": 1.0,
        "limit:rotX:physics:high": -1.0,
        "limit:rotY:physics:low": 1.0,
        "limit:rotY:physics:high": -1.0,
        "forbidden_applied_schemas": [
            "PhysicsLimitAPI:rotZ", "PhysicsDriveAPI:rotZ"
        ],
        "rotZ_limit_or_drive_properties_must_be_absent": True,
        "joint_z_collinear_with_plug_datum_B_local_z": True,
    }:
        raise ValueError("nut/body D6 bearing attribute contract changed")
    if authored.get("robot_articulation") != {
        "newton:selfCollisionEnabled": True,
    }:
        raise ValueError("robot self-collision authored attribute changed")
    if solver.get("overlay_precedence") != (
        "explicitly_authored_physx_collision_values_win_in_local_schema"
    ):
        raise ValueError("local PhysX/Newton overlay precedence is not frozen")
    if solver.get("resolved_api_readback_status") != "REQUIRED_DURING_A2_AND_A3":
        raise ValueError("solver resolved readback must remain an A2/A3 hard gate")
    required_readback = {
        "family", "prim_path", "collider_index", "typeName", "appliedSchemas",
        "collisionEnabled", "local_bounds", "world_bounds_at_canonical_pose",
        "geometry_type", "physics_approximation", "closed_manifold",
        "positive_volume", "convex", "topology_signature", "materialRole",
        "responseRole", "effective_physics_material_binding",
        "material_binding_source_prim", "resolved_compliant_stiffness_n_m",
        "resolved_compliant_damping_n_s_m", "resolved_accelerationSpring",
        "nearest_rigid_body_owner", "owner_rigidBodyEnabled",
        "owner_kinematicEnabled", "offset_class", "contactOffset_m",
        "restOffset_m", "collision_group_memberships", "filteredPairs_sources",
        "pass",
    }
    if set(solver.get("resolved_readback_required_fields", [])) != required_readback:
        raise ValueError("resolved collider readback field inventory changed")
    required_property_readback = {
        "semantic_id", "prim_path", "readback_kind", "property_name",
        "expected_value", "resolved_value", "hasAuthoredOpinion",
        "relationship_targets", "pass",
    }
    if set(solver.get("resolved_property_readback_required_fields", [])) != required_property_readback:
        raise ValueError("resolved property readback field inventory changed")
    required_family_pair_readback = {
        "left_family", "right_family", "decision_rule_id", "expected_decision",
        "resolved_decision", "expected_response_class",
        "expected_left_member_count", "expected_right_member_count",
        "resolved_left_member_count", "resolved_right_member_count",
        "expected_concrete_leaf_pair_count", "resolved_concrete_leaf_pair_count",
        "compliant_side_count", "collision_group_sources",
        "filteredPairs_sources", "ancestor_filter_sources",
        "joint_collision_gate", "matched_final_rule_count", "pass",
    }
    if set(solver.get("resolved_family_pair_readback_required_fields", [])) != required_family_pair_readback:
        raise ValueError("resolved family-pair readback field inventory changed")
    required_filter_source_readback = {
        "source_prim_path", "source_schema", "source_property_or_relationship",
        "affected_left_family", "affected_right_family", "decision_rule_id",
        "expected_effect", "resolved_effect",
        "expected_concrete_leaf_pair_count", "resolved_concrete_leaf_pair_count",
        "pass",
    }
    if set(solver.get("resolved_filter_source_row_required_fields", [])) != required_filter_source_readback:
        raise ValueError("resolved filter-source readback field inventory changed")
    result_contract = _mapping(
        solver.get("resolved_readback_result_contract"),
        "resolved readback result contract",
    )
    if (
        result_contract.get("schema_version")
        != "kcg_d38999_keyed_physical_r11_resolved_readback_v3"
        or result_contract.get("generator_id")
        != "kcg_d38999_keyed_v2_composed_stage_reader_v2"
        or result_contract.get("contract_revision")
        != "d38999_keyed_v2_r11_a0_family_algebra_v1"
        or result_contract.get("required_top_level_fields") != [
            "schema_version", "generator_id", "contract_revision", "asset_path",
            "root_prim", "collider_rows", "property_rows", "family_pair_rows",
            "filter_source_rows", "summary",
        ]
        or result_contract.get("production_release_API_accepts_composed_asset_path_only") is not True
        or result_contract.get("caller_supplied_expected_inventory_allowed") is not False
        or result_contract.get("caller_supplied_actual_mapping_is_candidate_only_not_release_evidence") is not True
        or result_contract.get("expected_inventory_authority")
        != "validator_internal_frozen_blueprint_and_solver_contract"
        or result_contract.get("expected_collider_row_count") != 15037
        or result_contract.get("expected_property_row_count") != 22
        or result_contract.get("expected_family_pair_row_count") != 406
        or result_contract.get("expected_filter_source_row_count") != 387
        or result_contract.get("collider_row_identity_fields")
        != ["family", "prim_path", "collider_index"]
        or result_contract.get("property_row_identity_fields")
        != ["semantic_id", "prim_path", "property_name"]
        or result_contract.get("family_pair_row_identity_fields")
        != ["left_family", "right_family"]
        or result_contract.get("filter_source_row_identity_fields")
        != ["source_prim_path", "source_schema", "source_property_or_relationship", "affected_left_family", "affected_right_family"]
        or result_contract.get("complete_expected_collider_inventory_required") is not True
        or result_contract.get("complete_expected_property_inventory_required") is not True
        or result_contract.get("complete_family_pair_inventory_required") is not True
        or result_contract.get("complete_filter_source_inventory_required") is not True
        or result_contract.get("every_row_must_pass") is not True
        or result_contract.get("expected_asset_path_must_equal_authorized_A2_output") is not True
        or result_contract.get("expected_root_prim") != SUCCESSOR_ROOT_PRIM
        or not isinstance(result_contract.get("required_summary_counts"), Mapping)
        or any(value != 0 for value in result_contract["required_summary_counts"].values())
    ):
        raise ValueError("resolved readback result schema changed")
    compliant_pairs = _mapping(
        solver.get("compliant_pair_readback_requirements"),
        "compliant pair readback requirements",
    )
    if compliant_pairs != {
        "allowed_pair_owner_classes": [
            "static_without_rigid_body_ancestor",
            "dynamic_owner_resolved_world_fixed_by_enabled_fixed_joint_chain",
            "dynamic_dynamic_internal_mechanism",
        ],
        "fixed_receptacle_contacts_require_world_fixed_load_path": True,
        "dynamic_dynamic_internal_pairs": ["anti_decoupling_detent"],
        "nearest_rigid_body_owner_required_on_each_nonstatic_side": True,
        "owner_rigidBodyEnabled_must_resolve_true": True,
        "owner_kinematicEnabled_must_resolve_false": True,
        "world_fixed_path_joint_type": "PhysicsFixedJoint",
        "world_fixed_path_every_joint_enabled": True,
        "collision_enabled_on_both_sides": True,
        "intended_pair_not_filtered": True,
    }:
        raise ValueError("compliant pair owner/load-path readback contract changed")
    joint_collision = _mapping(
        solver.get("helical_and_detent_pair_joint_collision_requirement"),
        "joint collision requirement",
    )
    if (
        joint_collision.get("physics_collisionEnabled_must_resolve_true") is not True
        or joint_collision.get("no_joint_subtree_default_may_silence_an_intended_pair")
        is not True
    ):
        raise ValueError("intended thread/detent pairs may be silenced by a joint")
    _validate_force_proxies(document)


def _proxy_range(
    parent: Mapping[str, Any], field: str, label: str
) -> tuple[float, float, float]:
    values = _mapping(parent.get(field), f"{label}.{field}")
    minimum = _positive(values.get("min"), f"{label}.{field}.min")
    nominal = _positive(values.get("nominal"), f"{label}.{field}.nominal")
    maximum = _positive(values.get("max"), f"{label}.{field}.max")
    if not minimum <= nominal <= maximum:
        raise ValueError(f"{label}.{field} nominal is outside its A3 range")
    return minimum, nominal, maximum


def _validate_compliant_role(
    role: Mapping[str, Any], label: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if role.get("compliant_contact_acceleration_spring") is not False:
        raise ValueError(f"{label} must use a force-based compliant contact")
    stiffness = _proxy_range(role, "stiffness_n_m", label)
    damping = _proxy_range(role, "damping_n_s_m", label)
    return stiffness, damping


def _validate_force_proxies(document: Mapping[str, Any]) -> None:
    boundaries = _mapping(
        document.get("physical_proxy_boundaries"), "physical_proxy_boundaries"
    )
    if boundaries.get("source_kind") != "FROZEN_SIMULATION_PROXY":
        raise ValueError("physical proxy boundaries lost their source classification")
    mass = _mapping(boundaries.get("mass_policy"), "mass policy")
    for field, expected in {
        "loose_plug_body_mass_kg": 0.23,
        "coupling_nut_mass_kg": 0.08,
        "fixed_receptacle_mass_kg": 0.30,
        "fixture_mass_kg": 5.0,
    }.items():
        if _positive(mass.get(field), f"mass policy {field}") != expected:
            raise ValueError(f"mass policy {field} changed")
    if (
        mass.get("origin") != "inherited_proxy_target_mass_not_hardware_truth"
        or mass.get("center_of_mass_and_inertia")
        != "explicit_frozen_axisymmetric_analytic_proxies_not_collision_geometry_union"
        or mass.get("overlapping_contact_and_collision_proxy_volume_contributes_to_mass")
        is not False
        or mass.get("validate_positive_definite_and_triangle_inequality") is not True
    ):
        raise ValueError("mass/inertia proxy boundary changed")
    roles = _mapping(boundaries.get("force_parameters"), "force_parameters")
    if frozenset(roles) != REQUIRED_FORCE_PROXY_ROLES:
        raise ValueError("force-transmitting proxy role inventory is incomplete")

    contacts = _mapping(
        roles.get("socket_contact_sleeves_61"), "socket contact sleeves"
    )
    if (
        contacts.get("representation")
        != "six_fixed_annular_wedge_petals_per_labeled_socket_with_compliant_contact"
        or contacts.get("petal_count_per_socket") != 6
        or contacts.get("socket_count") != 61
        or contacts.get("parameter_scope") != "per_petal"
        or contacts.get("preload_representation")
        != "authored_nominal_geometric_overlap"
        or contacts.get("force_curve_claim") != "simulation_proxy_only"
    ):
        raise ValueError("61 socket sleeve collision representation changed")
    contact_stiffness, contact_damping = _validate_compliant_role(
        contacts, "socket contact sleeves"
    )
    if (
        (contact_stiffness[0], contact_stiffness[2]) != (2000.0, 8000.0)
        or (contact_damping[0], contact_damping[2]) != (0.05, 0.20)
    ):
        raise ValueError("socket contact A3 calibration bounds changed")
    if _positive(
        contacts.get("nominal_pin_diameter_m"), "nominal pin diameter"
    ) != 0.001016:
        raise ValueError("size-20 nominal pin diameter changed")
    if _positive(
        contacts.get("nominal_radial_interference_m"),
        "socket nominal radial interference",
    ) != 0.0000127:
        raise ValueError("socket nominal radial interference changed")
    if _positive(
        contacts.get("maximum_physical_compliant_deflection_m"),
        "socket maximum physical compliant deflection",
    ) != 0.000075:
        raise ValueError("socket compliant deflection limit changed")
    socket_interference = _positive(
        contacts.get("nominal_radial_interference_m"),
        "socket nominal radial interference",
    )
    for field, expected in (
        (
            "nominal_per_socket_aggregate_stiffness_n_m",
            6.0 * contact_stiffness[1],
        ),
        (
            "nominal_per_socket_aggregate_damping_n_s_m",
            6.0 * contact_damping[1],
        ),
        (
            "nominal_per_socket_radial_preload_force_n",
            6.0 * contact_stiffness[1] * socket_interference,
        ),
    ):
        declared = _positive(contacts.get(field), f"socket {field}")
        if not math.isclose(declared, expected, abs_tol=1.0e-12):
            raise ValueError(f"socket aggregate {field} is inconsistent")

    fingers = _mapping(roles.get("shell_spring_fingers"), "shell spring fingers")
    if (
        fingers.get("representation")
        != "twelve_fixed_compliant_contact_segments_with_radial_preload"
        or fingers.get("segment_count") != 12
        or fingers.get("parameter_scope") != "per_segment"
        or fingers.get("preload_representation")
        != "authored_nominal_geometric_overlap"
        or fingers.get("material_role") != "spring_finger"
        or fingers.get("force_curve_claim") != "simulation_proxy_only"
    ):
        raise ValueError("shell spring-finger representation changed")
    finger_stiffness, finger_damping = _validate_compliant_role(
        fingers, "shell spring fingers"
    )
    if (
        (finger_stiffness[0], finger_stiffness[2]) != (6000.0, 24000.0)
        or (finger_damping[0], finger_damping[2]) != (0.5, 3.0)
    ):
        raise ValueError("shell spring-finger A3 calibration bounds changed")
    finger_preload = _positive(
        fingers.get("nominal_radial_preload_m"), "spring-finger preload"
    )
    if finger_preload != 0.000080:
        raise ValueError("spring-finger preload geometry changed")
    if _positive(
        fingers.get("required_lead_before_first_electrical_contact_m"),
        "spring-finger lead",
    ) != 0.00102:
        raise ValueError("spring-finger event lead changed")
    finger_force = _mapping(
        fingers.get("complete_connector_axial_force_acceptance_n"),
        "shell spring-finger force acceptance",
    )
    if (
        _positive(finger_force.get("min"), "spring-finger force minimum") != 2.0
        or _positive(finger_force.get("max"), "spring-finger force maximum")
        != 156.0
        or fingers.get("complete_connector_force_range_source_kind")
        != "PUBLIC_SPEC_VERIFIED_TABLE_VII_SHELL_24_25"
    ):
        raise ValueError("shell-25 complete-connector spring-finger force range changed")
    for field, expected in (
        ("nominal_aggregate_radial_stiffness_n_m", 12.0 * finger_stiffness[1]),
        ("nominal_aggregate_damping_n_s_m", 12.0 * finger_damping[1]),
        (
            "nominal_aggregate_radial_preload_force_n",
            12.0 * finger_stiffness[1] * finger_preload,
        ),
    ):
        if not math.isclose(
            _positive(fingers.get(field), f"spring fingers {field}"), expected,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"spring-finger aggregate {field} is inconsistent")

    barriers = _mapping(
        roles.get("interfacial_pin_barriers_61"), "interfacial pin barriers"
    )
    if (
        barriers.get("representation")
        != "sixty_one_fixed_axisymmetric_compliant_barrier_colliders"
        or barriers.get("barrier_count") != 61
        or barriers.get("angular_wedge_count_per_barrier") != 24
        or barriers.get("axial_profile_band_count_per_barrier") != 2
        or barriers.get("parameter_scope")
        != "per_barrier_effective_distributed_equally_across_24_angular_wedges"
        or barriers.get("paired_hard_geometry")
        != "sixty_one_size20_socket_entry_annulus_chamfers"
        or barriers.get("preload_representation")
        != "compression_created_by_mating_from_zero_at_barrier_touch_plane"
        or barriers.get("compliant_deflection_definition")
        != "resolved_contact_normal_overlap_not_post_contact_axial_travel"
        or barriers.get("axial_travel_must_not_be_used_as_spring_deflection")
        is not True
        or barriers.get("material_role") != "interfacial_pin_barrier"
        or barriers.get("force_curve_claim") != "simulation_proxy_only"
        or barriers.get("linearized_force_reference_is_not_an_A3_pass_value")
        is not True
        or _positive(
            barriers.get("unreached_profile_geometric_overlap_cap_m"),
            "pin-barrier unreached profile cap",
        )
        != 0.000600
        or barriers.get("unreached_profile_cap_is_not_force_or_energy_input")
        is not True
        or barriers.get("every_profile_band_of_one_angular_wedge_uses_same_response_material")
        is not True
        or barriers.get("aggregate_reference_multiplier_is_angular_wedges_not_profile_pieces")
        is not True
    ):
        raise ValueError("61 interfacial pin-barrier representation changed")
    barrier_stiffness, barrier_damping = _validate_compliant_role(
        barriers, "interfacial pin barriers"
    )
    if (
        (barrier_stiffness[0], barrier_stiffness[2]) != (125.0, 500.0)
        or (barrier_damping[0], barrier_damping[2]) != (0.010, 0.040)
    ):
        raise ValueError("pin-barrier A3 calibration bounds changed")
    wedge_stiffness = _proxy_range(
        barriers,
        "authored_per_angular_wedge_stiffness_n_m",
        "interfacial pin-barrier wedge",
    )
    wedge_damping = _proxy_range(
        barriers,
        "authored_per_angular_wedge_damping_n_s_m",
        "interfacial pin-barrier wedge",
    )
    for aggregate, wedge, label in (
        (barrier_stiffness, wedge_stiffness, "stiffness"),
        (barrier_damping, wedge_damping, "damping"),
    ):
        if any(
            not math.isclose(24.0 * wedge[index], aggregate[index], abs_tol=1.0e-12)
            for index in range(3)
        ):
            raise ValueError(f"pin-barrier per-wedge {label} is not exactly 1/24")
    barrier_deflection = _proxy_range(
        barriers, "physical_normal_deflection_range_m", "interfacial pin barriers"
    )
    if barrier_deflection != (0.000255, 0.000295, 0.000335):
        raise ValueError("pin-barrier derived deflection range changed")
    if _positive(
        barriers.get("nominal_first_touch_datum_B_separation_m"),
        "pin-barrier first-touch separation",
    ) != 0.014305:
        raise ValueError("pin-barrier first-touch proxy changed")
    if _positive(
        barriers.get("nominal_post_contact_axial_travel_to_bottoming_m"),
        "pin-barrier post-contact axial travel",
    ) != 0.000745:
        raise ValueError("pin-barrier axial travel changed")
    for field, expected in (
        ("nominal_aggregate_stiffness_n_m", 61.0 * barrier_stiffness[1]),
        ("nominal_aggregate_damping_n_s_m", 61.0 * barrier_damping[1]),
    ):
        if not math.isclose(
            _positive(barriers.get(field), f"pin barriers {field}"), expected,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"pin-barrier aggregate {field} is inconsistent")
    if not math.isclose(
        _positive(
            barriers.get("nominal_tip_throat_linearized_force_reference_n"),
            "pin-barrier linearized force reference",
        ),
        61.0 * barrier_stiffness[1] * barrier_deflection[1],
        abs_tol=1.0e-12,
    ):
        raise ValueError("pin-barrier linearized reference is inconsistent")

    seals = _mapping(roles.get("peripheral_seal"), "peripheral seal")
    if (
        seals.get("representation")
        != "segmented_annular_compliant_contact_surfaces"
        or seals.get("segment_count") != 24
        or seals.get("parameter_scope") != "per_segment"
        or seals.get("preload_representation")
        != "compression_created_by_mating_from_zero_at_A_plane"
        or seals.get("series_I_or_II_0p61m_requirement_applied_to_series_III")
        is not False
        or seals.get("material_role") != "peripheral_seal"
        or seals.get("force_curve_claim") != "simulation_proxy_only"
    ):
        raise ValueError("peripheral seal representation changed")
    seal_stiffness, seal_damping = _validate_compliant_role(seals, "peripheral seal")
    if (
        (seal_stiffness[0], seal_stiffness[2]) != (400.0, 1600.0)
        or (seal_damping[0], seal_damping[2]) != (0.2, 2.0)
    ):
        raise ValueError("peripheral seal A3 calibration bounds changed")
    seal_deflection = _proxy_range(
        seals, "physical_deflection_range_m", "peripheral seal"
    )
    if seal_deflection != (0.000280, 0.000435, 0.000590):
        raise ValueError("Series-III seal deflection range changed")
    if seals.get("deflection_derivation") != (
        "bottoming_proxy_separation_minus_public_A_static_seal_first_touch"
    ):
        raise ValueError("seal deflection derivation changed")
    for field, expected in (
        ("nominal_aggregate_stiffness_n_m", 24.0 * seal_stiffness[1]),
        ("nominal_aggregate_damping_n_s_m", 24.0 * seal_damping[1]),
        (
            "nominal_aggregate_preload_force_at_bottoming_n",
            24.0 * seal_stiffness[1] * seal_deflection[1],
        ),
    ):
        if not math.isclose(
            _positive(seals.get(field), f"seals {field}"), expected,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"seal aggregate {field} is inconsistent")

    detent = _mapping(
        roles.get("anti_decoupling_detent"), "anti-decoupling detent"
    )
    if (
        detent.get("representation")
        != "continuous_analytic_base_cylinder_plus_36_convex_tooth_prisms_and_three_compliant_followers"
        or detent.get("rejected_r8_representation")
        != "1368_separately_closed_annular_wedges_with_false_internal_partition_faces"
        or detent.get("rejected_r8_representation_may_not_be_reused") is not True
        or detent.get("tooth_count") != 36
        or detent.get("follower_count") != 3
        or detent.get("parameter_scope") != "per_follower"
        or detent.get("preload_representation")
        != "authored_nominal_geometric_overlap"
        or detent.get("material_role") != "anti_decoupling_detent"
    ):
        raise ValueError("physical anti-decoupling geometry changed")
    detent_stiffness, detent_damping = _validate_compliant_role(
        detent, "anti-decoupling detent"
    )
    if (
        detent_stiffness != (110000.0, 110000.0, 110000.0)
        or detent_damping != (2.0, 2.0, 2.0)
    ):
        raise ValueError("detent per-follower A3 calibration bounds changed")
    radius = _positive(detent.get("mean_radius_m"), "detent mean radius")
    preload = _positive(detent.get("nominal_radial_preload_m"), "detent preload")
    rise = _positive(detent.get("cam_radial_rise_m"), "detent cam rise")
    maximum_deflection = _positive(
        detent.get("maximum_radial_deflection_m"), "detent maximum deflection"
    )
    forward_angle = _positive(
        detent.get("forward_ramp_angle_deg"), "detent forward ramp angle"
    )
    reverse_angle = _positive(
        detent.get("reverse_face_angle_deg"), "detent reverse face angle"
    )
    if (
        radius != 0.022
        or preload != 0.000001
        or rise != 0.000050
        or maximum_deflection != 0.000051
        or forward_angle != 8.0
        or reverse_angle != 55.0
        or not math.isclose(preload + rise, maximum_deflection, abs_tol=1.0e-12)
    ):
        raise ValueError("detent cam/preload geometry changed")
    forward_span = math.degrees(rise / (radius * math.tan(math.radians(forward_angle))))
    reverse_span = math.degrees(rise / (radius * math.tan(math.radians(reverse_angle))))
    declared_forward_span = _positive(
        detent.get("forward_ramp_angular_span_deg"), "detent forward span"
    )
    declared_reverse_span = _positive(
        detent.get("reverse_face_angular_span_deg"), "detent reverse span"
    )
    declared_dwell = _positive(
        detent.get("remaining_dwell_per_tooth_deg"), "detent dwell"
    )
    if (
        not math.isclose(declared_forward_span, forward_span, abs_tol=1.0e-6)
        or not math.isclose(declared_reverse_span, reverse_span, abs_tol=1.0e-6)
        or not math.isclose(
            declared_dwell, 10.0 - forward_span - reverse_span, abs_tol=1.0e-6
        )
    ):
        raise ValueError("detent tooth profile is inconsistent with its angles")
    forward = _proxy_range(
        detent, "initial_forward_component_proxy_target_nm", "anti-decoupling detent"
    )
    reverse = _mapping(
        detent.get("reverse_component_proxy_envelope_nm"),
        "anti-decoupling reverse proxy envelope",
    )
    reverse_minimum_initial = _positive(
        reverse.get("minimum_initial"), "reverse proxy minimum initial"
    )
    reverse_nominal_initial = _positive(
        reverse.get("nominal_initial"), "reverse proxy nominal initial"
    )
    reverse_nominal_full_ramp = _positive(
        reverse.get("nominal_full_ramp"), "reverse proxy nominal full ramp"
    )
    reverse_maximum_allowed = _positive(
        reverse.get("maximum_allowed"), "reverse proxy maximum allowed"
    )
    if (
        reverse_maximum_allowed != 0.55
        or detent.get("complete_pair_public_torque_limits_are_not_detent_limits")
        is not True
    ):
        raise ValueError("detent proxy was confused with complete-pair torque limits")
    tan_forward = math.tan(math.radians(forward_angle))
    minimum_initial_torque = (
        3.0 * detent_stiffness[0] * preload * tan_forward * radius
    )
    nominal_initial_torque = (
        3.0 * detent_stiffness[1] * preload * tan_forward * radius
    )
    maximum_forward_torque = (
        3.0 * detent_stiffness[2] * maximum_deflection * tan_forward * radius
    )
    if (
        not math.isclose(minimum_initial_torque, forward[0], abs_tol=1.0e-12)
        or not math.isclose(nominal_initial_torque, forward[1], abs_tol=1.0e-5)
        or not math.isclose(nominal_initial_torque, forward[2], abs_tol=1.0e-12)
        or _positive(
            detent.get("full_forward_ramp_component_proxy_max_nm"),
            "detent full-ramp maximum",
        ) != 0.06
        or maximum_forward_torque
        > _positive(
            detent.get("full_forward_ramp_component_proxy_max_nm"),
            "detent full-ramp maximum",
        )
        + 1.0e-12
    ):
        raise ValueError("detent stiffness/preload cannot meet its forward torque budget")
    nominal_reverse_torque = (
        3.0
        * detent_stiffness[1]
        * preload
        * math.tan(math.radians(reverse_angle))
        * radius
    )
    minimum_reverse_torque = (
        3.0
        * detent_stiffness[0]
        * preload
        * math.tan(math.radians(reverse_angle))
        * radius
    )
    nominal_full_ramp_reverse_torque = (
        3.0
        * detent_stiffness[1]
        * maximum_deflection
        * math.tan(math.radians(reverse_angle))
        * radius
    )
    if (
        not math.isclose(
            reverse_minimum_initial, minimum_reverse_torque, abs_tol=1.0e-12
        )
        or not math.isclose(
            reverse_nominal_initial, nominal_reverse_torque, abs_tol=1.0e-12
        )
        or not math.isclose(
            reverse_nominal_full_ramp,
            nominal_full_ramp_reverse_torque,
            abs_tol=1.0e-12,
        )
        or nominal_full_ramp_reverse_torque > reverse_maximum_allowed
    ):
        raise ValueError("detent reverse torque is inconsistent with current nominal")
    if not math.isclose(
        _positive(
            detent.get("nominal_aggregate_initial_normal_force_n"),
            "detent aggregate initial normal force",
        ),
        3.0 * detent_stiffness[1] * preload,
        abs_tol=1.0e-12,
    ):
        raise ValueError("detent aggregate normal force is inconsistent")
    if detent.get("directional_hysteresis_claim") != (
        "physical_geometry_proxy_not_hardware_curve"
    ):
        raise ValueError("detent proxy was promoted to hardware truth")

    thread = _mapping(roles.get("coupling_thread"), "coupling thread proxy")
    if (
        thread.get("representation")
        != "three_start_segmented_helical_rail_and_follower_collision"
        or thread.get("start_phases_deg") != [0.0, 120.0, 240.0]
        or thread.get("runtime_engagement_switch_allowed") is not False
        or thread.get("software_axial_pose_write_allowed") is not False
    ):
        raise ValueError("physical three-start thread proxy changed")
    if (
        _positive(thread.get("pitch_m"), "thread proxy pitch") != 0.00254
        or _positive(thread.get("lead_m_per_revolution"), "thread proxy lead")
        != 0.00762
    ):
        raise ValueError("thread proxy pitch or lead changed")
    bearing = _mapping(boundaries.get("nut_body_bearing"), "nut/body bearing")
    if (
        bearing.get("representation")
        != "generic_D6_cylindrical_bearing_proxy_with_physical_shoulders"
        or bearing.get("joint_type") != "UsdPhysics.Joint"
        or bearing.get("joint_prim_suffix") != "/LoosePlug/CouplingNutJoint"
        or bearing.get("body0_prim_suffix") != "/LoosePlug/BodyAssembly"
        or bearing.get("body1_prim_suffix") != "/LoosePlug/CouplingNut"
        or bearing.get("locked_degrees_of_freedom")
        != ["transX", "transY", "rotX", "rotY"]
        or bearing.get("free_degree_of_freedom") != "rotZ"
        or bearing.get("limited_degree_of_freedom") != "transZ"
        or bearing.get("physics_collision_enabled") is not True
        or bearing.get("shoulder_contacts_before_joint_backup_limit") is not True
        or bearing.get("joint_drive_allowed") is not False
        or bearing.get("joint_friction_substitute_for_detent_allowed") is not False
    ):
        raise ValueError("nut/body cylindrical bearing architecture changed")
    if bearing.get("body_child_transforms_relative_to_loose_plug") != {
        "BodyAssembly": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "CouplingNut": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }:
        raise ValueError("plug body/nut child transforms changed")
    joint_frame = _mapping(bearing.get("joint_frame"), "nut/body joint frame")
    if joint_frame != {
        "local_pos0_m": [0.0, 0.0, 0.020],
        "local_pos1_m": [0.0, 0.0, 0.020],
        "local_rot0_wxyz": [1.0, 0.0, 0.0, 0.0],
        "local_rot1_wxyz": [1.0, 0.0, 0.0, 0.0],
        "joint_z_collinear_with_plug_datum_B_local_z": True,
    }:
        raise ValueError("nut/body D6 joint frame or axis changed")
    backup = _mapping(
        bearing.get("transZ_joint_backup_limits_m"), "nut/body backup limits"
    )
    shoulder = _mapping(
        bearing.get("physical_shoulder_contact_endplay_m"), "shoulder endplay"
    )
    if (
        (_finite(backup.get("low"), "backup low"), _finite(backup.get("high"), "backup high"))
        != (-0.00015, 0.00015)
        or (
            _finite(shoulder.get("low"), "shoulder low"),
            _finite(shoulder.get("high"), "shoulder high"),
        )
        != (-0.00005, 0.00005)
        or not backup["low"] < shoulder["low"] < shoulder["high"] < backup["high"]
    ):
        raise ValueError("physical shoulders do not contact before D6 backup limits")
    shoulder_geometry = _mapping(
        bearing.get("physical_shoulder_collision_geometry"),
        "physical shoulder collision geometry",
    )
    if (
        shoulder_geometry.get("representation")
        != "four_annular_ring_colliders_two_opposed_stop_pairs"
        or _positive(
            shoulder_geometry.get("radial_inner_m"), "shoulder inner radius"
        )
        != 0.0175
        or _positive(
            shoulder_geometry.get("radial_outer_m"), "shoulder outer radius"
        )
        != 0.0196
        or _positive(
            shoulder_geometry.get("axial_thickness_m"), "shoulder thickness"
        )
        != 0.00030
        or shoulder_geometry.get("annulus_convex_wedge_segment_count") != 48
        or shoulder_geometry.get("capped_solid_cylinder_allowed") is not False
        or shoulder_geometry.get("collider_owner_must_match_named_body") is not True
        or shoulder_geometry.get(
            "contact_planes_must_be_derived_from_collision_geometry_bounds"
        )
        is not True
        or shoulder_geometry.get("custom_metadata_only_contact_plane_forbidden")
        is not True
        or shoulder_geometry.get("collision_filter_policy")
        != "only_named_opposed_stop_pairs_enabled_between_body0_and_body1"
        or shoulder_geometry.get("material_role")
        != "coupling_bearing_and_shoulder"
    ):
        raise ValueError("physical shoulder collider blueprint changed")
    positive_stop = _mapping(
        shoulder_geometry.get("positive_transZ_stop"), "positive shoulder stop"
    )
    negative_stop = _mapping(
        shoulder_geometry.get("negative_transZ_stop"), "negative shoulder stop"
    )
    if positive_stop != {
        "body0_collider_suffix": (
            "/LoosePlug/BodyAssembly/NutBearingShoulders/PositiveStop"
        ),
        "body1_collider_suffix": (
            "/LoosePlug/CouplingNut/NutBearingShoulders/PositiveStop"
        ),
        "body0_neutral_contact_surface_local_z_m": 0.03000,
        "body1_neutral_contact_surface_local_z_m": 0.02995,
        "body0_contact_face_bound": "minZ",
        "body1_contact_face_bound": "maxZ",
        "body0_collider_center_local_z_m": 0.03015,
        "body1_collider_center_local_z_m": 0.02980,
        "expected_contact_transZ_m": 0.00005,
    } or negative_stop != {
        "body0_collider_suffix": (
            "/LoosePlug/BodyAssembly/NutBearingShoulders/NegativeStop"
        ),
        "body1_collider_suffix": (
            "/LoosePlug/CouplingNut/NutBearingShoulders/NegativeStop"
        ),
        "body0_neutral_contact_surface_local_z_m": 0.01000,
        "body1_neutral_contact_surface_local_z_m": 0.01005,
        "body0_contact_face_bound": "maxZ",
        "body1_contact_face_bound": "minZ",
        "body0_collider_center_local_z_m": 0.00985,
        "body1_collider_center_local_z_m": 0.01020,
        "expected_contact_transZ_m": -0.00005,
    }:
        raise ValueError("physical shoulder paths or geometry-derived planes changed")
    if not math.isclose(
        positive_stop["body0_neutral_contact_surface_local_z_m"]
        - positive_stop["body1_neutral_contact_surface_local_z_m"],
        positive_stop["expected_contact_transZ_m"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        negative_stop["body0_neutral_contact_surface_local_z_m"]
        - negative_stop["body1_neutral_contact_surface_local_z_m"],
        negative_stop["expected_contact_transZ_m"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("physical shoulder contact plane arithmetic changed")
    thickness = shoulder_geometry["axial_thickness_m"]
    if not math.isclose(
        positive_stop["body0_collider_center_local_z_m"] - 0.5 * thickness,
        positive_stop["body0_neutral_contact_surface_local_z_m"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        positive_stop["body1_collider_center_local_z_m"] + 0.5 * thickness,
        positive_stop["body1_neutral_contact_surface_local_z_m"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        negative_stop["body0_collider_center_local_z_m"] + 0.5 * thickness,
        negative_stop["body0_neutral_contact_surface_local_z_m"],
        abs_tol=1.0e-12,
    ) or not math.isclose(
        negative_stop["body1_collider_center_local_z_m"] - 0.5 * thickness,
        negative_stop["body1_neutral_contact_surface_local_z_m"],
        abs_tol=1.0e-12,
    ):
        raise ValueError("physical shoulder collider centers would overlap at neutral")
    accounting = _mapping(
        boundaries.get("deformation_and_energy_accounting"),
        "deformation and energy accounting",
    )
    if (
        accounting.get("compliant_deflection_is_not_counted_against_hard_contact_penetration_limit")
        is not True
        or accounting.get("compliant_exclusion_requires_resolved_intended_collider_pair")
        is not True
        or accounting.get("compliant_exclusion_requires_effective_material_binding")
        is not True
        or accounting.get("compliant_exclusion_requires_positive_resolved_stiffness")
        is not True
        or accounting.get("missing_or_wrong_binding_is_noncompliant_hard_contact")
        is not True
        or _positive(
            accounting.get("hard_contact_gap_or_penetration_limit_m"),
            "hard penetration limit",
        )
        != 0.00005
        or accounting.get("initial_preload_energy_reference")
        != "after_gravity_free_preload_settle_before_external_probe"
        or accounting.get("external_work_must_be_included_in_energy_balance")
        is not True
        or accounting.get("energy_state_equation")
        != "E=K+Ug+sum(0.5*k_eff*deflection^2)"
        or accounting.get("energy_residual_equation")
        != "R=delta_E-W_external+E_dissipation"
        or accounting.get("unexplained_energy_gain_definition") != "max(0,R)"
        or accounting.get("required_energy_trace_terms")
        != [
            "kinetic_energy_j",
            "gravitational_potential_energy_j",
            "compliant_energy_by_role_j",
            "external_applied_work_j",
            "kinematic_or_constraint_work_j",
            "friction_dissipation_j",
            "damping_dissipation_j",
            "residual_j",
        ]
        or accounting.get("every_material_scale_resettles_and_rebases_t0") is not True
        or _positive(
            accounting.get("unexplained_energy_gain_j_max"),
            "unexplained energy gain",
        )
        != 0.000001
    ):
        raise ValueError("hard penetration, compliant deflection, or energy accounting changed")
    if not isinstance(boundaries.get("passive_energy_rationale"), str):
        raise ValueError("force proxies lack a passive-energy rationale")
    calibration = _mapping(
        boundaries.get("a3_calibration_policy"), "A3 calibration policy"
    )
    if calibration != {
        "tunable_nominal_fields": [
            "force_parameters.socket_contact_sleeves_61.stiffness_n_m.nominal",
            "force_parameters.socket_contact_sleeves_61.damping_n_s_m.nominal",
            "force_parameters.shell_spring_fingers.stiffness_n_m.nominal",
            "force_parameters.shell_spring_fingers.damping_n_s_m.nominal",
            "force_parameters.interfacial_pin_barriers_61.stiffness_n_m.nominal",
            "force_parameters.interfacial_pin_barriers_61.damping_n_s_m.nominal",
            "force_parameters.peripheral_seal.stiffness_n_m.nominal",
            "force_parameters.peripheral_seal.damping_n_s_m.nominal",
            "force_parameters.anti_decoupling_detent.stiffness_n_m.nominal",
            "force_parameters.anti_decoupling_detent.damping_n_s_m.nominal",
        ],
        "range_endpoints_and_all_geometry_are_frozen_during_a3": True,
        "dependent_aggregate_and_torque_fields_must_be_recomputed": True,
        "a2_geometry_or_topology_change_allowed_during_a3": False,
        "selected_nominals_freeze_in_new_contract_revision_at_a5": True,
        "realized_contact_constraint_calibration": {
            "authored_material_stiffness_is_an_initial_guess_not_proof_of_effective_force_slope": True,
            "geometry_or_topology_change_allowed": False,
            "record_active_collider_pairs_and_contact_points_per_physical_element_at_every_sample": True,
            "resolve_force_and_overlap_in_each_contact_normal_frame": True,
            "target_effective_normal_force_slope_n_m": {
                "socket_petal_per_petal": 4000.0,
                "spring_finger_per_finger": 12000.0,
                "pin_barrier_per_barrier": 250.0,
                "peripheral_seal_per_segment": 800.0,
                "detent_follower_per_follower": 110000.0,
            },
            "target_relative_error_max": 0.10,
            "force_curve_must_be_nonnegative_monotonic_and_finite_across_declared_deflection_range": True,
            "band_or_seam_constraint_count_changes_must_not_create_force_discontinuity": True,
            "force_discontinuity_relative_to_full_scale_max": 0.05,
            "only_declared_k_and_d_nominals_may_be_tuned_within_frozen_ranges": True,
        },
    }:
        raise ValueError("A3 calibration field boundary changed")
    if boundaries.get("tuning_boundary") != (
        "only_explicit_a3_calibration_policy_nominals_may_change_within_frozen_ranges"
    ):
        raise ValueError("force-proxy A3/A5 tuning boundary changed")


def _validate_self_collision_filter(document: Mapping[str, Any]) -> None:
    policy = _mapping(document.get("self_collision_filter"), "self_collision_filter")
    if policy.get("policy") != "topology_adjacent_only_initially":
        raise ValueError("self-collision filtering must start adjacent-only")
    if (
        policy.get("successor_robot_asset_must_be_regenerated_from_source") is not True
        or policy.get("editing_existing_physics_usda_in_place_allowed") is not False
        or policy.get("exclusion_authoring_api")
        != "UsdPhysics.FilteredPairsAPI_on_semantic_rigid_links"
    ):
        raise ValueError("successor robot self-collision must be regenerated and semantic")
    if policy.get("sampled_never_pairs_authorized") is not False:
        raise ValueError("sampled Never pairs cannot be promoted to exclusions")
    raw_pairs = policy.get("excluded_pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("self_collision_filter.excluded_pairs must be a list")
    normalized = []
    for index, pair in enumerate(raw_pairs):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) and item for item in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError(f"self-collision excluded pair {index} is invalid")
        normalized.append(tuple(sorted(pair)))
    if len(normalized) != len(set(normalized)):
        raise ValueError("self-collision excluded pairs contain duplicates")
    if frozenset(normalized) != REQUIRED_SELF_COLLISION_EXCLUSIONS:
        raise ValueError("self-collision exclusions differ from 16 adjacent pairs")
    requirements = set(policy.get("additional_exclusion_requirements", []))
    if requirements != {
        "visual_collision_overlay_proves_permanent_joint_overlap",
        "exact_pair_reason_recorded",
        "deliberate_positive_control_still_detected",
        "full_path_and_perturbation_envelope_passed",
    }:
        raise ValueError("additional self-collision exclusion gate changed")


def _xml_vector(element: ET.Element | None, attribute: str, default: str) -> list[float]:
    text = default if element is None else element.get(attribute, default)
    values = [float(value) for value in text.split()]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid source XML vector {attribute}={text!r}")
    return values


def _source_xml_inventory(path_text: str) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    path = (WORKSPACE_ROOT / path_text).resolve()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot parse frozen robot source {path_text}: {exc}") from exc
    links = {element.get("name"): element for element in root.findall("link")}
    joints = {element.get("name"): element for element in root.findall("joint")}
    if None in links or None in joints:
        raise ValueError(f"unnamed link or joint in frozen robot source {path_text}")
    return links, joints


def _same_float_list(actual: list[float], expected: Any, tolerance: float = 1.0e-12) -> bool:
    return (
        isinstance(expected, list)
        and len(actual) == len(expected)
        and all(math.isclose(a, float(b), abs_tol=tolerance) for a, b in zip(actual, expected))
    )


def _validate_robot_source_xml_against_blueprint(blueprint: Mapping[str, Any]) -> None:
    """Keep the no-digest A0 robot blueprint tied to its declared Xacro values."""

    source = _mapping(blueprint.get("source_chain"), "robot source chain")
    aggregate_path = (WORKSPACE_ROOT / str(source["aggregate_xacro"])).resolve()
    try:
        aggregate_root = ET.parse(aggregate_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot parse frozen aggregate Xacro: {exc}") from exc
    aggregate_includes = [
        element.get("filename")
        for element in aggregate_root.iter()
        if element.tag.rsplit("}", 1)[-1] == "include"
    ]
    if aggregate_root.get("name") != "handarm" or aggregate_includes != [
        "iiwa14.xacro", "hand.xacro"
    ]:
        raise ValueError("aggregate handarm Xacro include chain changed")
    expected_include_paths = [
        (aggregate_path.parent / include).resolve() for include in aggregate_includes
    ]
    if expected_include_paths != [
        (WORKSPACE_ROOT / str(source["arm_xacro"])).resolve(),
        (WORKSPACE_ROOT / str(source["hand_xacro"])).resolve(),
    ]:
        raise ValueError("aggregate Xacro no longer resolves to the frozen arm/hand sources")

    export_path = (WORKSPACE_ROOT / str(source["export_script"])).resolve()
    import_path = (WORKSPACE_ROOT / str(source["import_script"])).resolve()
    try:
        export_text = export_path.read_text(encoding="utf-8")
        import_text = import_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read frozen robot export/import source: {exc}") from exc
    if (
        "get_package_share_directory" in export_text
        or "kcg_moveit1" in export_text
        or '_AUTHORITATIVE_XACRO = _DESCRIPTION_PACKAGE_ROOT / "urdf" / "handarm.urdf.xacro"'
        not in export_text
        or 'if output_path.exists():' not in export_text
        or "refusing to overwrite exported URDF" not in export_text
    ):
        raise ValueError("robot exporter is not bound to the frozen workspace aggregate")
    if (
        'if not arguments.allow_self_collision:' not in import_text
        or 'if output_directory.exists():' not in import_text
        or 'output_directory.mkdir(parents=True, exist_ok=False)' not in import_text
        or 'allow_self_collision=arguments.allow_self_collision' not in import_text
    ):
        raise ValueError("robot importer lost its no-overwrite or self-collision gate")

    source_documents: dict[str, tuple[dict[str, ET.Element], dict[str, ET.Element]]] = {}
    for key in ("arm_xacro", "hand_xacro"):
        path_text = str(source[key])
        source_documents[path_text] = _source_xml_inventory(path_text)

    collisions = _mapping(blueprint.get("collision_inventory"), "robot collisions")
    mass_inventory = _mapping(
        blueprint.get("mass_property_inventory"), "robot mass properties"
    )
    mass_by_link = {
        row["link"]: row
        for row in mass_inventory["per_link_values"]
    }
    for collision_row in collisions["per_link_source_inventory"]:
        link_name = collision_row["link"]
        mass_row = mass_by_link[link_name]
        links, _ = source_documents[mass_row["source"]]
        link = links.get(link_name)
        if link is None:
            raise ValueError(f"frozen robot source lost link {link_name}")
        source_collisions = link.findall("collision")
        if len(source_collisions) != 1:
            raise ValueError(f"{link_name} must keep exactly one source collision")
        collision = source_collisions[0]
        if not _same_float_list(
            _xml_vector(collision.find("origin"), "xyz", "0 0 0"),
            collisions["every_collision_local_translation_m"],
        ) or not _same_float_list(
            _xml_vector(collision.find("origin"), "rpy", "0 0 0"),
            collisions["every_collision_local_rotation_rpy_rad"],
        ):
            raise ValueError(f"{link_name} source collision origin changed")
        mesh = collision.find("geometry/mesh")
        if mesh is None or mesh.get("filename") != collision_row["mesh_uri"]:
            raise ValueError(f"{link_name} source collision mesh changed")
        if not _same_float_list(
            _xml_vector(mesh, "scale", "1 1 1"), collisions["every_mesh_scale"]
        ):
            raise ValueError(f"{link_name} source collision scale changed")

        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"{link_name} source inertial is missing")
        mass_element = inertial.find("mass")
        inertia_element = inertial.find("inertia")
        if mass_element is None or inertia_element is None:
            raise ValueError(f"{link_name} source mass or inertia is missing")
        source_mass = float(mass_element.get("value", "nan"))
        source_com = _xml_vector(inertial.find("origin"), "xyz", "0 0 0")
        source_inertia = [
            float(inertia_element.get(name, "nan"))
            for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ]
        if (
            not math.isclose(source_mass, float(mass_row["mass_kg"]), abs_tol=1.0e-12)
            or not _same_float_list(source_com, mass_row["com_m"])
            or not _same_float_list(source_inertia, mass_row["inertia_six_kg_m2"])
        ):
            raise ValueError(f"{link_name} source mass, COM, or inertia changed")

    joints = _mapping(blueprint.get("joint_property_inventory"), "robot joints")
    for expected in joints["revolute_joints_exactly"]:
        _, source_joints = source_documents[expected["source"]]
        joint = source_joints.get(expected["joint"])
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"frozen robot source lost revolute joint {expected['joint']}")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        mimic = joint.find("mimic")
        if (
            parent is None or parent.get("link") != expected["parent"]
            or child is None or child.get("link") != expected["child"]
            or not _same_float_list(_xml_vector(origin, "xyz", "0 0 0"), expected["xyz_m"])
            or not _same_float_list(_xml_vector(origin, "rpy", "0 0 0"), expected["rpy_rad"])
            or not _same_float_list(_xml_vector(axis, "xyz", "1 0 0"), expected["axis"])
            or limit is None
            or not math.isclose(float(limit.get("lower", "nan")), float(expected["lower_rad"]), abs_tol=1.0e-12)
            or not math.isclose(float(limit.get("upper", "nan")), float(expected["upper_rad"]), abs_tol=1.0e-12)
            or not math.isclose(float(limit.get("effort", "nan")), float(expected["effort"]), abs_tol=1.0e-12)
            or not math.isclose(float(limit.get("velocity", "nan")), float(expected["velocity_rad_s"]), abs_tol=1.0e-12)
            or not math.isclose(float("0" if dynamics is None else dynamics.get("damping", "0")), float(expected["damping"]), abs_tol=1.0e-12)
            or not math.isclose(float("0" if dynamics is None else dynamics.get("friction", "0")), float(expected["friction"]), abs_tol=1.0e-12)
        ):
            raise ValueError(f"{expected['joint']} source transform or dynamics changed")
        expected_mimic = expected["mimic"]
        if expected_mimic is None:
            if mimic is not None:
                raise ValueError(f"active joint {expected['joint']} became a mimic")
        elif (
            mimic is None
            or mimic.get("joint") != expected_mimic["joint"]
            or not math.isclose(float(mimic.get("multiplier", "1")), float(expected_mimic["multiplier"]), abs_tol=1.0e-12)
            or not math.isclose(float(mimic.get("offset", "0")), float(expected_mimic["offset_rad"]), abs_tol=1.0e-12)
        ):
            raise ValueError(f"{expected['joint']} source mimic mapping changed")

    fixed_sources = {
        "world_iiwa_joint": str(source["arm_xacro"]),
        "hand2arm": str(source["hand_xacro"]),
    }
    for expected in joints["fixed_joints_exactly"]:
        _, source_joints = source_documents[fixed_sources[expected["joint"]]]
        joint = source_joints.get(expected["joint"])
        if joint is None or joint.get("type") != "fixed":
            raise ValueError(f"frozen robot source lost fixed joint {expected['joint']}")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if (
            parent is None or parent.get("link") != expected["source_parent"]
            or child is None or child.get("link") != expected["child"]
            or not _same_float_list(_xml_vector(origin, "xyz", "0 0 0"), expected["xyz_m"])
            or not _same_float_list(_xml_vector(origin, "rpy", "0 0 0"), expected["rpy_rad"])
        ):
            raise ValueError(f"{expected['joint']} source fixed transform changed")

    _, hand_joints = source_documents[str(source["hand_xacro"])]
    tcp_joint = hand_joints.get("handbase_to_grasp_tcp")
    if (
        tcp_joint is None
        or tcp_joint.get("type") != "fixed"
        or tcp_joint.find("parent") is None
        or tcp_joint.find("parent").get("link") != "handbase_link"
        or tcp_joint.find("child") is None
        or tcp_joint.find("child").get("link") != "grasp_tcp"
        or not _same_float_list(
            _xml_vector(tcp_joint.find("origin"), "xyz", "0 0 0"),
            [0.0, 0.0, 0.400],
        )
        or not _same_float_list(
            _xml_vector(tcp_joint.find("origin"), "rpy", "0 0 0"),
            [0.0, 0.0, 0.0],
        )
    ):
        raise ValueError("grasp_tcp source transform changed")


def _validate_realized_robot_hand_fixture_blueprint(
    document: Mapping[str, Any],
) -> None:
    blueprint = _mapping(
        document.get("realized_robot_hand_fixture_blueprint"),
        "realized_robot_hand_fixture_blueprint",
    )
    if set(blueprint) != {
        "status", "source_kind", "successor_robot_asset", "source_chain",
        "collision_inventory", "mass_property_inventory", "joint_property_inventory",
        "self_collision",
        "robot_material_partition", "semantic_frames_and_sensors",
        "fixture_load_path",
    }:
        raise ValueError("robot/hand/fixture authoring blueprint inventory changed")
    if (
        blueprint.get("status")
        != "FROZEN_FOR_MECHANICAL_A2_AUTHORING_AFTER_GLOBAL_A0_RELEASE"
        or blueprint.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("robot/hand/fixture blueprint release boundary changed")
    successor = _mapping(blueprint.get("successor_robot_asset"), "successor robot")
    if successor != {
        "output_path": "artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda",
        "overwrite_existing": False,
        "default_prim": "/handarm",
        "runtime_reference_root": "/World/HandArm",
        "articulation_path": "/World/HandArm/Geometry/world",
    }:
        raise ValueError("successor robot asset identity changed")
    source = _mapping(blueprint.get("source_chain"), "robot source chain")
    if source != {
        "aggregate_xacro": "src/iiwa_description/urdf/handarm.urdf.xacro",
        "arm_xacro": "src/iiwa_description/urdf/iiwa14.xacro",
        "hand_xacro": "src/iiwa_description/urdf/hand.xacro",
        "export_script": "src/kcg_connector/kcg_connector/export_isaac_urdf.py",
        "import_script": "src/kcg_connector/isaac/import_robot.py",
        "importer_cli_args_exactly": ["--allow-self-collision"],
        "importer": {
            "merge_fixed_joints": False,
            "merge_mesh": False,
            "collision_from_visuals": False,
            "allow_self_collision": True,
            "fix_base": True,
        },
    }:
        raise ValueError("robot source/import chain changed")
    collisions = _mapping(blueprint.get("collision_inventory"), "robot collisions")
    _require_exact_fields(
        collisions,
        {
            "expected_semantic_link_count": 17,
            "expected_collision_prim_count": 17,
            "source_collision_count_per_link": 1,
            "every_collision_local_translation_m": [0.0, 0.0, 0.0],
            "every_collision_local_rotation_rpy_rad": [0.0, 0.0, 0.0],
            "every_mesh_scale": [1.0, 1.0, 1.0],
            "every_usd_approximation": "convexHull",
            "every_collision_typeName": "Mesh",
            "every_collision_applied_schema": "PhysicsCollisionAPI",
            "every_collision_enabled": True,
            "every_collision_offset_class": "general",
            "every_collision_contactOffset_m": 0.00005,
            "every_collision_restOffset_m": 0.0,
            "nearest_rigid_body_owner_is_same_semantic_link": True,
            "every_owner_rigidBodyEnabled": True,
            "every_owner_kinematicEnabled": False,
            "every_owner_CCD_enabled": True,
            "effective_material_binding_required": True,
            "source_STL_points_and_triangle_topology_readback_required": True,
            "source_STL_vs_realized_mesh_comparison_uses_transformed_numeric_vertices_not_metadata": True,
            "automatic_convex_decomposition_or_visual_fallback_allowed": False,
            "extra_collision_prim_count_allowed": 0,
            "required_custom_attribute_names": {
                "source_mesh_uri": "kcg:sourceMeshUri",
                "material_role": "kcg:materialRole",
                "response_role": "kcg:responseRole",
            },
            "generated_deep_collider_paths_are_readback_results_not_A0_identity": True,
        },
        "robot collisions",
    )
    expected_meshes = {
        "iiwa_link_0": ("package://iiwa_description/meshes/iiwa14/collision/link_0_s.stl", "robot_structure"),
        "iiwa_link_1": ("package://iiwa_description/meshes/iiwa14/collision/link_1_s.stl", "robot_structure"),
        "iiwa_link_2": ("package://iiwa_description/meshes/iiwa14/collision/link_2_s.stl", "robot_structure"),
        "iiwa_link_3": ("package://iiwa_description/meshes/iiwa14/collision/link_3_s.stl", "robot_structure"),
        "iiwa_link_4": ("package://iiwa_description/meshes/iiwa14/collision/link_4_s.stl", "robot_structure"),
        "iiwa_link_5": ("package://iiwa_description/meshes/iiwa14/collision/link_5_s.stl", "robot_structure"),
        "iiwa_link_6": ("package://iiwa_description/meshes/iiwa14/collision/link_6_s.stl", "robot_structure"),
        "iiwa_link_7": ("package://iiwa_description/meshes/iiwa14/collision/link_7_s.stl", "robot_structure"),
        "handbase_link": ("package://iiwa_description/meshes/hand/collision/handbase_link_convex.stl", "robot_structure"),
        "f1Link1": ("package://iiwa_description/meshes/hand/collision/f1Link1_convex.stl", "finger_structure"),
        "f1Link2": ("package://iiwa_description/meshes/hand/collision/f1Link2_convex.stl", "finger_structure"),
        "f1Link3": ("package://iiwa_description/meshes/hand/collision/f1Link3_convex.stl", "fingertip_pad"),
        "f2Link1": ("package://iiwa_description/meshes/hand/collision/f2Link1_convex.stl", "finger_structure"),
        "f2Link2": ("package://iiwa_description/meshes/hand/collision/f2Link2_convex.stl", "fingertip_pad"),
        "f3Link1": ("package://iiwa_description/meshes/hand/collision/f3Link1_convex.stl", "finger_structure"),
        "f3Link2": ("package://iiwa_description/meshes/hand/collision/f3Link2_convex.stl", "finger_structure"),
        "f3Link3": ("package://iiwa_description/meshes/hand/collision/f3Link3_convex.stl", "fingertip_pad"),
    }
    raw_meshes = collisions.get("per_link_source_inventory")
    if not isinstance(raw_meshes, list):
        raise ValueError("per-link collision inventory must be a list")
    mesh_rows = {
        _mapping(row, "collision inventory row").get("link"): row for row in raw_meshes
    }
    if set(mesh_rows) != set(expected_meshes) or len(mesh_rows) != len(raw_meshes):
        raise ValueError("17-link collision source inventory changed")
    for link, (uri, role) in expected_meshes.items():
        if mesh_rows[link] != {
            "link": link, "mesh_uri": uri, "material_role": role,
            "response_role": "hard_rigid",
        }:
            raise ValueError(f"collision source or role changed for {link}")

    mass_inventory = _mapping(
        blueprint.get("mass_property_inventory"), "robot mass properties"
    )
    _require_exact_fields(
        mass_inventory,
        {
            "expected_mass_api_count": 17,
            "extra_mass_api_count_allowed": 0,
            "mass_api_owner_is_same_semantic_link": True,
            "every_mass_owner_rigidBodyEnabled": True,
            "every_mass_owner_kinematicEnabled": False,
            "expected_total_mass_kg": 24.591942,
            "inertia_six_order": ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"],
            "source_tensor_frame": "semantic_link_frame_at_declared_COM",
            "compare_reconstructed_full_tensor_in_semantic_link_frame": True,
            "raw_usd_diagonalInertia_tuple_equality_required": False,
            "raw_usd_principalAxes_tuple_equality_required": False,
            "positive_definite_and_triangle_inequality_required": True,
            "readback_comparison": {
                "mass_abs_tolerance_kg": 0.000001,
                "com_component_abs_tolerance_m": 0.0000001,
                "full_inertia_component_abs_tolerance_kg_m2": 0.000000001,
                "full_inertia_component_rel_tolerance": 0.00001,
                "usd_principal_axes_quaternion_order": "wxyz",
                "normalize_principal_axes_quaternion_before_reconstruction": True,
                "quaternion_sign_is_equivalent": True,
                "full_tensor_reconstruction_formula":
                "R_from_normalized_quaternion*diag(diagonalInertia)*transpose(R)",
                "compare_in_semantic_link_frame_after_COM_frame_rotation": True,
            },
        },
        "robot mass properties",
    )
    expected_mass_rows = {
        "iiwa_link_0": ("src/iiwa_description/urdf/iiwa14.xacro", 5.0, [-0.1, 0.0, 0.07], [0.05, 0.06, 0.03, 0.0, 0.0, 0.0]),
        "iiwa_link_1": ("src/iiwa_description/urdf/iiwa14.xacro", 4.0, [0.0, -0.03, 0.12], [0.1, 0.09, 0.02, 0.0, 0.0, 0.0]),
        "iiwa_link_2": ("src/iiwa_description/urdf/iiwa14.xacro", 4.0, [0.0003, 0.059, 0.042], [0.05, 0.018, 0.044, 0.0, 0.0, 0.0]),
        "iiwa_link_3": ("src/iiwa_description/urdf/iiwa14.xacro", 3.0, [0.0, 0.03, 0.13], [0.08, 0.075, 0.01, 0.0, 0.0, 0.0]),
        "iiwa_link_4": ("src/iiwa_description/urdf/iiwa14.xacro", 2.7, [0.0, 0.067, 0.034], [0.03, 0.01, 0.029, 0.0, 0.0, 0.0]),
        "iiwa_link_5": ("src/iiwa_description/urdf/iiwa14.xacro", 1.7, [0.0001, 0.021, 0.076], [0.02, 0.018, 0.005, 0.0, 0.0, 0.0]),
        "iiwa_link_6": ("src/iiwa_description/urdf/iiwa14.xacro", 1.8, [0.0, 0.0006, 0.0004], [0.005, 0.0036, 0.0047, 0.0, 0.0, 0.0]),
        "iiwa_link_7": ("src/iiwa_description/urdf/iiwa14.xacro", 0.3, [0.0, 0.0, 0.02], [0.001, 0.001, 0.001, 0.0, 0.0, 0.0]),
        "handbase_link": ("src/iiwa_description/urdf/hand.xacro", 1.3106, [0.0068536, -0.0032827, 0.15427], [0.0017579, 0.0017392, 0.0031039, 0.00003605, -0.0000062211, -0.0000056815]),
        "f1Link1": ("src/iiwa_description/urdf/hand.xacro", 0.1943, [-0.027217, 0.015636, -0.018412], [0.000050231, 0.000055471, 0.000035768, -0.0000044869, -0.00000082062, 0.00000057019]),
        "f1Link2": ("src/iiwa_description/urdf/hand.xacro", 0.073035, [-0.02357, -0.021421, 0.018942], [0.000023141, 0.000023914, 0.000044461, 0.000015948, 0.00000012581, 0.00000011946]),
        "f1Link3": ("src/iiwa_description/urdf/hand.xacro", 0.057879, [-0.016803, -0.020891, 0.011999], [0.000016904, 0.000012703, 0.000024528, 0.0000097694, 0.0000000091101, 0.000000012096]),
        "f2Link1": ("src/iiwa_description/urdf/hand.xacro", 0.073035, [-0.021392, -0.023597, -0.016858], [0.000026213, 0.000020842, 0.000044461, 0.000015725, 0.00000011369, 0.00000013105]),
        "f2Link2": ("src/iiwa_description/urdf/hand.xacro", 0.057879, [-0.012481, -0.023727, -0.012001], [0.000020429, 0.0000091773, 0.000024528, 0.0000082585, 0.0000000066195, 0.000000013619]),
        "f3Link1": ("src/iiwa_description/urdf/hand.xacro", 0.1943, [0.027217, 0.015636, 0.018412], [0.000050231, 0.000055471, 0.000035768, 0.0000044869, -0.00000082062, -0.00000057019]),
        "f3Link2": ("src/iiwa_description/urdf/hand.xacro", 0.073035, [-0.02357, -0.021421, 0.018942], [0.000023141, 0.000023914, 0.000044461, 0.000015948, 0.00000012581, 0.00000011946]),
        "f3Link3": ("src/iiwa_description/urdf/hand.xacro", 0.057879, [-0.016803, -0.020891, 0.011999], [0.000016904, 0.000012703, 0.000024528, 0.0000097694, 0.0000000091101, 0.000000012096]),
    }
    raw_mass_rows = mass_inventory.get("per_link_values")
    if not isinstance(raw_mass_rows, list):
        raise ValueError("per-link mass inventory must be a list")
    mass_rows = {
        _mapping(row, "mass inventory row").get("link"): row for row in raw_mass_rows
    }
    if set(mass_rows) != set(expected_mass_rows) or len(mass_rows) != len(raw_mass_rows):
        raise ValueError("17-link mass/inertia inventory changed")
    total_mass = 0.0
    for link, (source_path, mass, com, inertia) in expected_mass_rows.items():
        expected_row = {
            "link": link, "source": source_path, "mass_kg": mass,
            "com_m": com, "inertia_six_kg_m2": inertia,
        }
        if mass_rows[link] != expected_row:
            raise ValueError(f"mass, COM, or inertia changed for {link}")
        total_mass += _positive(mass_rows[link]["mass_kg"], f"{link} mass")
    if not math.isclose(total_mass, 24.591942, abs_tol=1.0e-12):
        raise ValueError("17-link total robot/hand mass changed")
    joints = _mapping(
        blueprint.get("joint_property_inventory"), "robot joint properties"
    )
    if (
        joints.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or joints.get("revolute_joint_count") != 15
        or joints.get("fixed_joint_count_in_physics_scope") != 2
        or joints.get("source_transform_convention")
        != "URDF_xyz_and_fixed_axis_RPY_with_R_equals_Rz_yaw_times_Ry_pitch_times_Rx_roll"
        or joints.get("usd_readback_transform_rule")
        != "reconstruct_body0_to_body1_zero_state_from_body_relationships_and_localPos0_localRot0_localPos1_localRot1_then_compare_to_source_transform"
        or joints.get("source_radians_to_usd_degrees_formula")
        != "source_rad*180/pi"
        or joints.get("drive_mapping") != {
            "required_applied_schemas": [
                "PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular",
                "PhysxJointAPI",
            ],
            "physics_jointEnabled": True,
            "physics_excludeFromArticulation": False,
            "drive:angular:physics:type": "force",
            "drive_angular_damping_equals_urdf_dynamics_damping": True,
            "drive_angular_maxForce_equals_urdf_limit_effort": True,
            "drive_angular_stiffness_authored": False,
            "drive_target_position_authored": False,
            "drive_target_velocity_authored": False,
            "physxJoint_maxJointVelocity_equals_urdf_velocity_rad_s_converted_to_deg_s": True,
            "physxJoint_jointFriction_equals_urdf_dynamics_friction": True,
            "runtime_drive_overlay_must_be_separately_recorded_and_may_not_change_asset_readback": True,
            "unexpected_revolute_or_D6_joint_count_allowed": 0,
            "unexpected_joint_state_or_drive_API_count_allowed": 0,
        }
        or joints.get("fixed_joint_readback") != {
            "physics_jointEnabled": True,
            "physics_excludeFromArticulation": False,
            "physics_collisionEnabled": False,
            "unexpected_fixed_joint_count_allowed": 0,
        }
        or joints.get("readback_tolerances") != {
            "local_position_abs_m": 0.0000001,
            "reconstructed_rotation_angle_abs_rad": 0.00001,
            "limit_abs_deg": 0.0001,
            "effort_abs": 0.0001,
            "velocity_abs_deg_s": 0.0001,
            "damping_abs": 0.000001,
            "friction_abs": 0.000001,
            "mimic_multiplier_abs": 0.000001,
            "mimic_offset_abs_rad": 0.000001,
        }
    ):
        raise ValueError("robot joint transform/property readback contract changed")
    joint_rows = joints.get("revolute_joints_exactly")
    if not isinstance(joint_rows, list) or len(joint_rows) != 15:
        raise ValueError("robot revolute-joint inventory must contain 15 rows")
    expected_joint_names = {
        *(f"iiwa_joint_{index}" for index in range(1, 8)),
        "f1j1", "f1j2", "f1j3", "f2j1", "f2j2",
        "f3j1", "f3j2", "f3j3",
    }
    if {row.get("joint") for row in joint_rows if isinstance(row, Mapping)} != expected_joint_names:
        raise ValueError("robot revolute-joint names changed")
    if joints.get("fixed_joints_exactly") != [
        {
            "joint": "world_iiwa_joint", "source_parent": "world",
            "realized_parent": "handarm",
            "child": "iiwa_link_0", "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
        {
            "joint": "hand2arm", "source_parent": "iiwa_link_ee",
            "realized_parent": "iiwa_link_ee",
            "child": "handbase_link", "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0],
        },
    ]:
        raise ValueError("robot fixed-joint inventory changed")
    self_collision = _mapping(
        blueprint.get("self_collision"), "realized self collision"
    )
    pairs = self_collision.get("normalized_undirected_pairs_exactly")
    if not isinstance(pairs, list):
        raise ValueError("realized self-collision pairs must be a list")
    normalized = frozenset(tuple(sorted(pair)) for pair in pairs)
    if (
        self_collision.get("articulation_attribute")
        != {"newton:selfCollisionEnabled": True}
        or self_collision.get("filter_api") != "UsdPhysics.FilteredPairsAPI"
        or self_collision.get("filter_owner") != "semantic_rigid_link"
        or normalized != REQUIRED_SELF_COLLISION_EXCLUSIONS
        or len(pairs) != 16
        or self_collision.get("additional_pair_count") != 0
        or self_collision.get("sampled_never_pair_count") != 0
    ):
        raise ValueError("realized self-collision authoring contract changed")
    partition = _mapping(
        blueprint.get("robot_material_partition"), "robot material partition"
    )
    expected_partition = {
        "robot_structure": [
            "iiwa_link_0", "iiwa_link_1", "iiwa_link_2", "iiwa_link_3",
            "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7",
            "handbase_link",
        ],
        "finger_structure": ["f1Link1", "f1Link2", "f2Link1", "f3Link1", "f3Link2"],
        "fingertip_pad": ["f1Link3", "f2Link2", "f3Link3"],
    }
    if (
        partition.get("material_values_are_immutable_through_A2_A3") is not True
        or partition.get("role_to_links") != expected_partition
        or partition.get("high_friction_link_set_exactly")
        != ["f1Link3", "f2Link2", "f3Link3"]
        or partition.get("asset_material_prim_root") != "/handarm/PhysicsMaterials"
        or partition.get("runtime_material_prim_root")
        != "/World/HandArm/PhysicsMaterials"
    ):
        raise ValueError("realized robot material partition changed")
    frames = _mapping(
        blueprint.get("semantic_frames_and_sensors"), "semantic frames and sensors"
    )
    handbase = "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link"
    if frames.get("handbase_path") != handbase:
        raise ValueError("canonical handbase path changed")
    tcp = _mapping(frames.get("grasp_tcp"), "realized grasp TCP")
    if tcp != {
        "path": f"{handbase}/grasp_tcp", "typeName": "Xform",
        "translation_m": [0.0, 0.0, 0.400],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "required_schema": "IsaacSiteAPI",
        "forbidden_schemas": ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"],
    }:
        raise ValueError("realized grasp TCP contract changed")
    wrist = _mapping(frames.get("wrist_ft"), "realized wrist FT")
    if wrist != {
        "sensor_prim_path": None, "joint_path": "/World/HandArm/Physics/hand2arm",
        "joint_type": "PhysicsFixedJoint", "body0_semantic": "iiwa_link_ee",
        "body1_semantic": "handbase_link",
        "local_translation_both_m": [0.0, 0.0, 0.0],
        "local_rotation_both_wxyz": [1.0, 0.0, 0.0, 0.0],
        "raw_frame": "handbase_link",
        "raw_semantics": "isaac_incoming_joint_reaction_on_child",
        "canonical_from_raw": "negate_all_six_components",
        "reaction_row_rule": "metadata_joint_index_plus_one",
        "independent_body_or_collider": "forbidden",
    }:
        raise ValueError("realized wrist reaction-wrench contract changed")
    cameras = _mapping(frames.get("cameras"), "realized cameras")
    if dict(cameras) != {
        "physical_housing_or_mass_or_collision": "forbidden",
        "duplicate_live_view_camera_prims_allowed": False,
        "canonical_authoring_owner":
        "runner_session_layer_after_robot_reference_before_physics_start",
        "successor_robot_asset_precontains_camera_prims": False,
        "authoring_must_fail_if_canonical_prim_already_exists": True,
        "camera_xform_or_intrinsics_write_after_physics_start_count_allowed": 0,
        "resolution_px": [1280, 720],
        "channels_exactly": ["rgb", "distance_to_image_plane"],
        "focal_length_mm": 24.0,
        "horizontal_aperture_mm": 20.955,
        "vertical_aperture_mm": 11.7871875,
        "clipping_range_m": [0.02, 10.0],
        "camera_cv_axes": ["x_right", "y_down", "z_forward"],
        "usd_camera_forward_axis": "minus_z",
        "palm": {
            "path": f"{handbase}/PalmCamera",
            "T_HC_cv": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.315],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "wrist": {
            "path": f"{handbase}/WristCamera",
            "T_HC_cv": [
                [0.9899494936611665, 0.0, 0.1414213562373095, -0.150],
                [0.0, 1.0, 0.0, 0.0],
                [-0.1414213562373095, 0.0, 0.9899494936611665, 0.060],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "forbidden_schemas": [
            "PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"
        ],
    }:
        raise ValueError("canonical zero-mass camera blueprint changed")
    fixture = _mapping(blueprint.get("fixture_load_path"), "realized fixture")
    _require_exact_fields(
        fixture,
        {
            "scene_root": "/World/D38999TabletopV1",
            "table_path": "/World/D38999TabletopV1/Table",
            "fixture_path": "/World/D38999TabletopV1/FixedFixture",
            "pair_reference_path": "/World/D38999TabletopV1/D38999Pair",
            "pair_model_root": "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3",
            "fixed_receptacle_path": "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3/FixedReceptacle",
            "fixture_to_world_joint_path": "/World/D38999TabletopV1/Joints/FixtureToWorld",
            "receptacle_to_fixture_joint_path": "/World/D38999TabletopV1/Joints/ReceptacleToFixture",
            "direct_receptacle_to_world": "forbidden",
        },
        "realized fixture",
    )
    if fixture.get("fixture_to_world") != {
        "body0_relationship_target_count": 0,
        "body0_semantic": "world",
        "body1": "/World/D38999TabletopV1/FixedFixture",
        "joint_type": "PhysicsFixedJoint", "enabled": True,
        "collision_enabled": False,
        "localPos0_m": [0.550, 0.185, 0.220],
        "localRot0_wxyz": [1.0, 0.0, 0.0, 0.0],
        "localPos1_m": [0.0, 0.0, 0.0],
        "localRot1_wxyz": [1.0, 0.0, 0.0, 0.0],
    } or fixture.get("receptacle_to_fixture") != {
        "body0": "/World/D38999TabletopV1/FixedFixture",
        "body1": "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3/FixedReceptacle",
        "joint_type": "PhysicsFixedJoint", "enabled": True,
        "collision_enabled": False,
        "localPos0_m": [0.0, 0.0, 0.052],
        "localRot0_wxyz": [0.0, 1.0, 0.0, 0.0],
        "localPos1_m": [0.0, 0.0, 0.0],
        "localRot1_wxyz": [1.0, 0.0, 0.0, 0.0],
    }:
        raise ValueError("realized fixture fixed-joint topology changed")
    if fixture.get("fixture_collision_geometry") != {
        "type_name": "Mesh",
        "representation": "explicit_metric_box_8_vertices_6_outward_quads",
        "local_points_m": [
            [-0.070, -0.070, -0.020],
            [0.070, -0.070, -0.020],
            [0.070, 0.070, -0.020],
            [-0.070, 0.070, -0.020],
            [-0.070, -0.070, 0.020],
            [0.070, -0.070, 0.020],
            [0.070, 0.070, 0.020],
            [-0.070, 0.070, 0.020],
        ],
        "face_vertex_counts": [4, 4, 4, 4, 4, 4],
        "face_vertex_indices": [
            0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
            1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7,
        ],
        "local_extent_m": [
            [-0.070, -0.070, -0.020],
            [0.070, 0.070, 0.020],
        ],
        "subdivision_scheme": "none",
        "collision_approximation": "convexHull",
        "transform_op_order": ["xformOp:translate"],
        "translation_m": [0.550, 0.185, 0.220],
        "reset_xform_stack": False,
        "authored_scale_op_count": 0,
        "rigid_body_owner_is_same_prim": True,
        "collision_api_required": True,
    }:
        raise ValueError("fixture unscaled metric collision geometry changed")
    mass_derivation = _mapping(
        fixture.get("connector_body_mass_derivation"), "connector mass derivation"
    )
    expected_mass_derivation = {
        "source_kind": "FROZEN_SIMULATION_PROXY",
        "method": "explicit_uniform_axisymmetric_analytic_proxy_per_rigid_owner",
        "collision_geometry_used_for_mass_derivation": False,
        "reason": "collision_and_force_proxy_solids_intentionally_overlap_or_duplicate_volume",
        "reference_frame": "each_connector_body_root",
        "principal_axes_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "bodies": {
            "FixedReceptacle": {
                "primitive": "uniform_solid_cylinder_aligned_local_z",
                "mass_kg": 0.30,
                "radius_m": 0.01961,
                "height_m": 0.01692,
                "local_com_m": [0.0, 0.0, 0.00846],
                "diagonal_inertia_kg_m2": [
                    0.0000359985675,
                    0.0000359985675,
                    0.000057682815,
                ],
            },
            "BodyAssembly": {
                "primitive": "uniform_solid_cylinder_aligned_local_z",
                "mass_kg": 0.23,
                "radius_m": 0.02220,
                "height_m": 0.03050,
                "local_com_m": [0.0, 0.0, 0.01525],
                "diagonal_inertia_kg_m2": [
                    0.0000461680916666667,
                    0.0000461680916666667,
                    0.0000566766,
                ],
            },
            "CouplingNut": {
                "primitive": "uniform_hollow_cylinder_aligned_local_z",
                "mass_kg": 0.08,
                "outer_radius_m": 0.02400,
                "inner_radius_m": 0.02090,
                "height_m": 0.03250,
                "local_com_m": [0.0, 0.0, 0.01425],
                "diagonal_inertia_kg_m2": [
                    0.0000272978666666667,
                    0.0000272978666666667,
                    0.0000405124,
                ],
            },
        },
        "explicit_PhysicsMassAPI_required": True,
        "PhysX_automatic_mass_allowed": False,
        "values_are_immutable_during_A2_through_A5": True,
    }
    if mass_derivation != expected_mass_derivation:
        raise ValueError("connector body mass-property derivation changed")
    for body_name, body in expected_mass_derivation["bodies"].items():
        mass_kg = float(body["mass_kg"])
        height_m = float(body["height_m"])
        if body["primitive"] == "uniform_solid_cylinder_aligned_local_z":
            radius_m = float(body["radius_m"])
            expected_ixy = mass_kg * (3.0 * radius_m**2 + height_m**2) / 12.0
            expected_izz = 0.5 * mass_kg * radius_m**2
        else:
            outer_radius_m = float(body["outer_radius_m"])
            inner_radius_m = float(body["inner_radius_m"])
            expected_ixy = mass_kg * (
                3.0 * (outer_radius_m**2 + inner_radius_m**2) + height_m**2
            ) / 12.0
            expected_izz = 0.5 * mass_kg * (
                outer_radius_m**2 + inner_radius_m**2
            )
        inertia = body["diagonal_inertia_kg_m2"]
        if not (
            math.isclose(float(inertia[0]), expected_ixy, rel_tol=0.0, abs_tol=1e-18)
            and math.isclose(float(inertia[1]), expected_ixy, rel_tol=0.0, abs_tol=1e-18)
            and math.isclose(float(inertia[2]), expected_izz, rel_tol=0.0, abs_tol=1e-18)
            and float(inertia[0]) + float(inertia[1]) >= float(inertia[2])
            and float(inertia[0]) + float(inertia[2]) >= float(inertia[1])
            and float(inertia[1]) + float(inertia[2]) >= float(inertia[0])
        ):
            raise ValueError(f"{body_name} analytic inertia no longer closes")
    if fixture.get("fixture_mass_properties") != {
        "mass_kg": 5.0,
        "local_com_m": [0.0, 0.0, 0.0],
        "diagonal_inertia_kg_m2": [
            0.0088333333333, 0.0088333333333, 0.0163333333333
        ],
        "derivation": "uniform_box_from_frozen_size",
    }:
        raise ValueError("fixture explicit mass-property blueprint changed")
    _validate_robot_source_xml_against_blueprint(blueprint)
    if blueprint != FROZEN_REALIZED_ROBOT_HAND_FIXTURE_BLUEPRINT:
        raise ValueError(
            "complete realized robot/hand/fixture blueprint differs from its "
            "independent frozen literal snapshot"
        )


def _validate_robot_and_fixture(document: Mapping[str, Any]) -> None:
    boundaries = _mapping(
        document.get("sensor_and_robot_boundaries"), "sensor_and_robot_boundaries"
    )
    inertia = _mapping(boundaries.get("robot_inertia"), "robot inertia")
    required_links = [
        "iiwa_link_0", "iiwa_link_1", "iiwa_link_2", "iiwa_link_3",
        "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7",
        "handbase_link", "f1Link1", "f1Link2", "f1Link3", "f2Link1",
        "f2Link2", "f3Link1", "f3Link2", "f3Link3",
    ]
    if (
        inertia.get("active_path") != "src/iiwa_description/urdf/handarm.urdf.xacro"
        or inertia.get("arm_source") != "src/iiwa_description/urdf/iiwa14.xacro"
        or inertia.get("hand_source") != "src/iiwa_description/urdf/hand.xacro"
        or inertia.get("source_kind") != "FROZEN_SIMULATION_PROXY"
        or inertia.get("alternate_local_files_authorized") is not False
        or inertia.get("validate_all_links_for_finite_positive_inertia") is not True
        or inertia.get("expected_physical_collision_link_count") != 17
        or inertia.get("required_collision_links") != required_links
        or inertia.get("mass_and_inertia_source_must_be_recorded_per_link") is not True
    ):
        raise ValueError("robot collision/mass/inertia source contract changed")
    mimic = _mapping(boundaries.get("hand_mimic"), "hand mimic")
    if (
        mimic.get("mode") != "ideal_one_to_one_joint_proxy"
        or mimic.get("source_kind") != "FROZEN_SIMULATION_PROXY"
    ):
        raise ValueError("hand mimic proxy changed")
    tcp = _mapping(boundaries.get("grasp_tcp"), "grasp TCP")
    if (
        tcp.get("mode") != "massless_noncolliding_virtual_task_frame"
        or tcp.get("canonical_path")
        != "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link/grasp_tcp"
        or tcp.get("typeName") != "Xform"
        or tcp.get("required_schema") != "IsaacSiteAPI"
        or tcp.get("handbase_translation_m") != [0.0, 0.0, 0.400]
        or tcp.get("handbase_rotation_wxyz") != [1.0, 0.0, 0.0, 0.0]
        or tcp.get("physical_body_exists") is not False
        or tcp.get("forbidden_applied_schemas")
        != ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
        or tcp.get("claim") != "simulation_control_frame_not_hardware_tool_center"
        or tcp.get("must_be_checked_against_terminal_grip_workspace_in_A2") is not True
    ):
        raise ValueError("grasp TCP proxy/frame boundary changed")
    fingers = _mapping(boundaries.get("finger_channels"), "finger channels")
    if (
        fingers.get("mode") != "physx_joint_effort_proxy"
        or fingers.get("fingertip_tactile_exists") is not False
    ):
        raise ValueError("finger joint-effort proxy was promoted to tactile sensing")
    wrist = _mapping(boundaries.get("wrist_ft"), "wrist FT")
    if (
        wrist.get("mode") != "hand2arm_fixed_joint_reaction_wrench_proxy"
        or wrist.get("physical_sensor_body_exists") is not False
        or wrist.get("source_joint_semantic_id") != "hand2arm"
        or wrist.get("canonical_joint_path") != "/World/HandArm/Physics/hand2arm"
        or wrist.get("joint_type") != "PhysicsFixedJoint"
        or wrist.get("body0_semantic") != "iiwa_link_ee"
        or wrist.get("body1_semantic") != "handbase_link"
        or wrist.get("joint_local_translation_both_m") != [0.0, 0.0, 0.0]
        or wrist.get("joint_local_rotation_both_wxyz") != [1.0, 0.0, 0.0, 0.0]
        or wrist.get("raw_frame") != "handbase_link"
        or wrist.get("raw_semantics") != "isaac_incoming_joint_reaction_on_child"
        or wrist.get("canonical_from_raw") != "negate_all_six_components"
        or wrist.get("reaction_row_rule") != "metadata_joint_index_plus_one"
        or wrist.get("independent_sensor_rigid_body_or_collider_forbidden")
        is not True
        or set(wrist.get("required_before_control", []))
        != {
            "frame", "sign", "reference_point", "tare", "gravity_compensation",
            "inertia_compensation", "timestamp",
        }
    ):
        raise ValueError("wrist reaction-wrench proxy boundary changed")
    cameras = _mapping(boundaries.get("cameras"), "cameras")
    if (
        cameras.get("mode") != "fixed_zero_mass_render_sensor_proxy"
        or cameras.get("physical_housing_exists") is not False
        or cameras.get("forbidden_applied_schemas")
        != ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
        or cameras.get("real_mount_calibration_claimed") is not False
        or cameras.get("extrinsics_must_be_frozen_before_a5") is not True
        or cameras.get("canonical_prim_suffixes") != ["/PalmCamera", "/WristCamera"]
        or cameras.get("duplicate_live_view_camera_prims_allowed") is not False
        or cameras.get("resolution_px") != [1280, 720]
        or cameras.get("focal_length_mm") != 24.0
        or cameras.get("horizontal_aperture_mm") != 20.955
        or cameras.get("vertical_aperture_mm") != 11.7871875
        or cameras.get("clipping_range_m") != [0.02, 10.0]
        or cameras.get("palm_T_HC_cv")
        != [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.315], [0.0, 0.0, 0.0, 1.0]]
        or cameras.get("wrist_T_HC_cv")
        != [[0.9899494936611665, 0.0, 0.1414213562373095, -0.150],
            [0.0, 1.0, 0.0, 0.0],
            [-0.1414213562373095, 0.0, 0.9899494936611665, 0.060],
            [0.0, 0.0, 0.0, 1.0]]
    ):
        raise ValueError("camera render-proxy boundary changed")

    fixture = _mapping(document.get("fixture_and_world_model"), "fixture/world model")
    if fixture.get("source_kind") != "FROZEN_SIMULATION_PROXY":
        raise ValueError("fixture/world model must remain a simulation proxy")
    table = _mapping(fixture.get("table"), "table model")
    fixture_body = _mapping(fixture.get("fixture"), "fixture model")
    load_path = _mapping(fixture.get("load_path"), "fixture load path")
    receptacle = _mapping(fixture.get("fixed_receptacle"), "fixed receptacle model")
    if (
        fixture.get("scene_root") != "/World/D38999TabletopV1"
        or table.get("path") != "/World/D38999TabletopV1/Table"
        or table.get("representation") != "world_static_collision_box"
        or table.get("center_m") != [0.550, 0.000, 0.160]
        or table.get("size_m") != [0.800, 0.900, 0.080]
        or table.get("material_role") != "table"
        or fixture_body.get("representation")
        != "dynamic_rigid_collision_box_fixed_to_world"
        or fixture_body.get("path") != "/World/D38999TabletopV1/FixedFixture"
        or fixture_body.get("center_m") != [0.550, 0.185, 0.220]
        or fixture_body.get("size_m") != [0.140, 0.140, 0.040]
        or _positive(fixture_body.get("mass_kg"), "fixture mass") != 5.0
        or fixture_body.get("local_com_m") != [0.0, 0.0, 0.0]
        or fixture_body.get("diagonal_inertia_kg_m2")
        != [0.0088333333333, 0.0088333333333, 0.0163333333333]
        or fixture_body.get("material_role") != "fixture_and_receptacle"
        or load_path.get("pair_reference_path") != "/World/D38999TabletopV1/D38999Pair"
        or load_path.get("pair_model_root")
        != "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3"
        or load_path.get("fixed_receptacle_path")
        != "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3/FixedReceptacle"
        or load_path.get("receptacle_to_fixture_joint_path")
        != "/World/D38999TabletopV1/Joints/ReceptacleToFixture"
        or load_path.get("fixture_to_world_joint_path")
        != "/World/D38999TabletopV1/Joints/FixtureToWorld"
        or load_path.get("receptacle_to_fixture_joint") != "UsdPhysics.FixedJoint"
        or load_path.get("fixture_to_world_joint") != "UsdPhysics.FixedJoint"
        or load_path.get("direct_receptacle_to_world_joint_allowed") is not False
        or load_path.get("direct_world_static_receptacle_allowed") is not False
        or load_path.get("fixed_joint_collision_enabled") is not False
        or load_path.get("fixture_to_world_frames") != {
            "body0_relationship_target_count": 0,
            "body0_semantic": "world",
            "body1": "/World/D38999TabletopV1/FixedFixture",
            "localPos0_m": [0.550, 0.185, 0.220],
            "localRot0_wxyz": [1.0, 0.0, 0.0, 0.0],
            "localPos1_m": [0.0, 0.0, 0.0],
            "localRot1_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        or load_path.get("receptacle_to_fixture_frames") != {
            "body0": "/World/D38999TabletopV1/FixedFixture",
            "body1": "/World/D38999TabletopV1/D38999Pair/D38999Shell25JKeyedPhysicalV3/FixedReceptacle",
            "localPos0_m": [0.0, 0.0, 0.052],
            "localRot0_wxyz": [0.0, 1.0, 0.0, 0.0],
            "localPos1_m": [0.0, 0.0, 0.0],
            "localRot1_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        or _positive(receptacle.get("mass_kg"), "fixed receptacle mass") != 0.30
        or receptacle.get("material_role") != "fixture_and_receptacle"
        or receptacle.get("infinite_fixture_stiffness_claim")
        != "simulation_proxy_only"
        or receptacle.get("mass_property_derivation")
        != "frozen_uniform_solid_cylinder_proxy_not_collision_geometry_union"
        or receptacle.get("local_com_m") != [0.0, 0.0, 0.00846]
        or receptacle.get("diagonal_inertia_kg_m2")
        != [0.0000359985675, 0.0000359985675, 0.000057682815]
        or receptacle.get("explicit_PhysicsMassAPI_required") is not True
        or receptacle.get("PhysX_automatic_mass_allowed") is not False
    ):
        raise ValueError("explicit receptacle-fixture-world physical load path changed")


def _validate_safety_and_firewall(document: Mapping[str, Any]) -> None:
    safety = _mapping(document.get("safety_contract"), "safety_contract")
    robot = _mapping(safety.get("robot_in_loop"), "robot_in_loop safety")
    component = _mapping(
        safety.get("connector_component_bench"), "component bench safety"
    )
    if _finite(robot.get("formal_perpendicular_moment_max_nm"), "robot moment") != 0.30:
        raise ValueError("robot formal perpendicular moment gate must remain 0.30 N*m")
    if robot.get("first_exceedance_fails_episode") is not True:
        raise ValueError("first formal moment exceedance must fail")
    if robot.get("may_be_relaxed_by_component_specification") is not False:
        raise ValueError("component torque limits cannot relax the robot gate")
    if (
        _finite(component.get("complete_pair_min_disengagement_nm"), "minimum disengagement") != 0.6
        or _finite(component.get("complete_pair_max_coupling_or_disengagement_nm"), "maximum coupling") != 4.6
        or component.get("limits_apply_to_complete_plug_receptacle_pair_not_detent_alone")
        is not True
        or component.get("may_run_without_robot_only") is not True
    ):
        raise ValueError("complete-pair public torque range changed")
    firewall = _mapping(document.get("truth_firewall"), "truth_firewall")
    required_forbidden = {
        "object_pose_truth",
        "physx_contact_points_normals_or_depth",
        "collider_identity",
        "contact_manifold",
        "thread_engagement_truth",
        "electrical_contact_truth",
        "nonexistent_fingertip_tactile",
    }
    if not required_forbidden.issubset(set(firewall.get("controller_forbidden_inputs", []))):
        raise ValueError("truth-firewall controller exclusions are incomplete")
    required_mechanisms = {
        "object_pose_write_after_start",
        "magnetic_attachment",
        "hidden_latch",
        "object_actuator",
        "software_written_axial_thread_progress",
    }
    if not required_mechanisms.issubset(set(firewall.get("forbidden_mechanisms", []))):
        raise ValueError("truth-firewall mechanism exclusions are incomplete")


def _validate_benches_phases_and_authorization(document: Mapping[str, Any]) -> None:
    if document.get("status") != "A0_FROZEN_A2_AUTHORIZED":
        raise ValueError(
            "this baseline contract only permits the frozen A0/A2-authoring state; "
            "later phase transitions require an independent release validator"
        )
    benches = _mapping(document.get("acceptance_benches"), "acceptance_benches")
    if benches.get("required_ids") != list(REQUIRED_BENCH_IDS):
        raise ValueError("P1-P14 acceptance matrix is incomplete or reordered")
    for bench_id in REQUIRED_BENCH_IDS:
        if not isinstance(benches.get(bench_id), str) or not benches[bench_id]:
            raise ValueError(f"{bench_id} lacks a description")
    phases = _mapping(document.get("phase_gates"), "phase_gates")
    if tuple(phases) != REQUIRED_PHASES:
        raise ValueError("phase_gates must contain ordered A0 through A5")
    final = dict(phases) == FINAL_PHASE_STATE
    a0 = _mapping(document.get("a0_source_freeze"), "a0_source_freeze")
    authorization = _mapping(document.get("authorization"), "authorization")
    if dict(phases) != {
        "A0": "FROZEN",
        "A1": "AUDIT_COMPLETE",
        "A2": "NOT_STARTED",
        "A3": "NOT_RUN",
        "A4": "NOT_REVIEWED",
        "A5": "NOT_FROZEN",
    }:
        raise ValueError("the frozen A0/A2-authoring phase state changed")
    if (
        a0.get("status") != "FROZEN"
        or a0.get("a2_asset_authoring_allowed") is not True
        or a0.get("unresolved_source_mappings") != []
    ):
        raise ValueError("the frozen A0 source-release latch changed")
    expected_a0_release_authorization = {
        "a2_asset_authoring_allowed": True,
        "connector_component_benches_allowed": False,
        "robot_in_loop_bench_allowed": False,
        "grasp_allowed": False,
        "camera_dataset_allowed": False,
        "visual_control_allowed": False,
        "insertion_allowed": False,
        "twist_allowed": False,
        "randomization_allowed": False,
        "training_allowed": False,
        "rl_allowed": False,
        "hardware_control_allowed": False,
    }
    if dict(authorization) != expected_a0_release_authorization:
        raise ValueError("the frozen A0 authorization latch changed")
    if bool(authorization.get("a2_asset_authoring_allowed")) != bool(
        a0.get("a2_asset_authoring_allowed")
    ):
        raise ValueError("A2 permission differs between A0 and authorization")
    if any(bool(authorization.get(field)) for field in DOWNSTREAM_AUTHORIZATION_FIELDS) and not final:
        raise ValueError("downstream work cannot be authorized before A0-A5 are final")
    if authorization.get("hardware_control_allowed") is not False:
        raise ValueError("hardware control is outside this simulation contract")
    if not final and any(
        bool(authorization.get(field))
        for field in (
            "connector_component_benches_allowed",
            "robot_in_loop_bench_allowed",
            *DOWNSTREAM_AUTHORIZATION_FIELDS,
        )
    ):
        # A later contract revision may open component benches after A2, but
        # this A0 release document remains stricter until independently
        # recomputed A2/A3 release objects are implemented and accepted.
        raise ValueError("the current A0 contract must remain globally fail-closed")


@dataclass(frozen=True)
class PhysicalModelContract:
    path: Path
    document: Mapping[str, Any]

    @property
    def phase_gates(self) -> Mapping[str, str]:
        return _mapping(self.document["phase_gates"], "phase_gates")

    @property
    def a2_asset_authoring_allowed(self) -> bool:
        return bool(self.document["authorization"]["a2_asset_authoring_allowed"])

    @property
    def downstream_authorized(self) -> bool:
        authorization = self.document["authorization"]
        return all(bool(authorization[field]) for field in DOWNSTREAM_AUTHORIZATION_FIELDS)

    @property
    def unresolved_a0_blockers(self) -> tuple[str, ...]:
        return tuple(
            str(item["id"])
            for item in self.document["a0_source_freeze"]["unresolved_source_mappings"]
            if item.get("blocking") is True
        )


def _validate_frozen_literal_snapshots(document: Mapping[str, Any]) -> None:
    immutable_sections = {
        key: document.get(key) for key in FROZEN_MODEL_IMMUTABLE_SECTIONS
    }
    if immutable_sections != FROZEN_MODEL_IMMUTABLE_SECTIONS:
        raise ValueError(
            "an immutable model/source/physics section differs from its "
            "independent frozen literal snapshot"
        )
    a0 = _mapping(document.get("a0_source_freeze"), "a0_source_freeze")
    if a0.get("resolved_decisions") != FROZEN_A0_RESOLVED_DECISIONS:
        raise ValueError("A0 resolved decisions differ from their frozen snapshot")
    if a0.get("resolved_source_mappings") != FROZEN_A0_RESOLVED_SOURCE_MAPPINGS:
        raise ValueError("A0 resolved source mappings differ from their frozen snapshot")


def load_physical_model_contract(
    path: Path | str = DEFAULT_CONTRACT_PATH,
) -> PhysicalModelContract:
    contract_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(contract_path.read_text(encoding="utf-8")), "document"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _reject_fingerprint_metadata(document)
    _validate_scope_and_evidence(document)
    _validate_identity(document)
    _validate_convex_cooking_representation(document)
    _validate_sources(document)
    _validate_a0(document)
    _validate_coordinate_contract(document)
    _validate_geometry(document)
    _validate_components(document)
    _validate_a2_collision_authoring_blueprint(document)
    _validate_materials_and_solver(document)
    _validate_self_collision_filter(document)
    _validate_realized_robot_hand_fixture_blueprint(document)
    _validate_robot_and_fixture(document)
    _validate_safety_and_firewall(document)
    _validate_benches_phases_and_authorization(document)
    _validate_frozen_literal_snapshots(document)
    return PhysicalModelContract(path=contract_path, document=document)


def safe_successor_asset_output(
    path: Path | str,
    contract: PhysicalModelContract | None = None,
) -> Path:
    model = contract or load_physical_model_contract()
    if not model.a2_asset_authoring_allowed:
        raise PermissionError(
            "A2 asset authoring is blocked until every A0 source mapping is frozen"
        )
    output = Path(path).expanduser().resolve()
    identity = model.document["identity"]
    expected = (
        WORKSPACE_ROOT
        / str(identity["recommended_asset_directory"])
        / str(identity["recommended_asset_name"])
    ).resolve()
    if output != expected:
        raise ValueError(f"successor asset output must be exactly {expected}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable asset: {output}")
    return output


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "PhysicalModelContract",
    "REQUIRED_BENCH_IDS",
    "REQUIRED_COMPONENT_IDS",
    "REQUIRED_FORCE_PROXY_ROLES",
    "REQUIRED_MATERIAL_ROLES",
    "REQUIRED_SELF_COLLISION_EXCLUSIONS",
    "REQUIRED_SEQUENCE_EVENTS",
    "REQUIRED_SEQUENCE_PRECEDENCE",
    "SCHEMA_VERSION",
    "SOURCE_KINDS",
    "SUCCESSOR_ASSET_NAME",
    "SUCCESSOR_REVISION",
    "SUCCESSOR_ROOT_PRIM",
    "SUCCESSOR_SCHEMA",
    "load_physical_model_contract",
    "safe_successor_asset_output",
]
