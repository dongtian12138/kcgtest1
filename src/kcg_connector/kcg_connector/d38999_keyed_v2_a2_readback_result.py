"""Fail-closed A2 composed-stage readback gate for the keyed D38999 r11 asset.

Expected collider identities, family-pair decisions, collision-group filters,
and D6-joint properties are derived internally from the frozen physical-model
contract. An external ``expected`` document is never an authority. Mapping
validation checks a reader candidate, but only the asset-path entry point may
produce release evidence. This module computes no file fingerprint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
import math
from numbers import Real
from pathlib import Path
import re
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence

import yaml

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    SUCCESSOR_ROOT_PRIM,
    WORKSPACE_ROOT,
    PhysicalModelContract,
    load_physical_model_contract,
)


SCHEMA_VERSION = "kcg_d38999_keyed_physical_r11_resolved_readback_v3"
GENERATOR_ID = "kcg_d38999_keyed_v2_composed_stage_reader_v2"
CONTRACT_REVISION = "d38999_keyed_v2_r11_a0_family_algebra_v1"
COLLECTION_NAMES = (
    "collider_rows",
    "property_rows",
    "family_pair_rows",
    "filter_source_rows",
)
FORBIDDEN_METADATA_KEY_PARTS = ("sha256", "checksum", "digest", "hash")
_RANGE_PREFIX = re.compile(r"^(\d+)\.\.(\d+)")
_HEX_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{32,}$")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [_mapping(row, f"{label}[{index}]") for index, row in enumerate(value)]


def _load_candidate(value: Mapping[str, Any] | Path | str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value).expanduser().resolve()
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "candidate")


def _walk_keys(value: Any, prefix: str = "document") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{prefix} contains a non-text key")
            path = f"{prefix}.{key}"
            yield path
            yield from _walk_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{index}]")


def _reject_fingerprint_metadata(document: Mapping[str, Any]) -> None:
    for path in _walk_keys(document):
        leaf = path.rsplit(".", 1)[-1].lower().replace("-", "_")
        if any(part in leaf for part in FORBIDDEN_METADATA_KEY_PARTS):
            raise ValueError(f"candidate contains forbidden fingerprint metadata at {path}")


def _identity(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[field] for field in fields)


def _require_exact_keys(row: Mapping[str, Any], required: set[str], label: str) -> None:
    actual = set(row)
    if actual != required:
        raise ValueError(
            f"{label} field inventory changed; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )


def _index_rows(
    rows: Sequence[Mapping[str, Any]], identity_fields: Sequence[str]
) -> tuple[dict[tuple[Any, ...], Mapping[str, Any]], int]:
    identities = [_identity(row, identity_fields) for row in rows]
    duplicate_count = sum(
        count - 1 for count in Counter(identities).values() if count > 1
    )
    indexed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for identity, row in zip(identities, rows):
        indexed.setdefault(identity, row)
    return indexed, duplicate_count


def _authorized_asset_path(model: PhysicalModelContract) -> Path:
    identity = model.document["identity"]
    return (
        WORKSPACE_ROOT
        / str(identity["recommended_asset_directory"])
        / str(identity["recommended_asset_name"])
    ).resolve()


def _owner_path(owner: str) -> str:
    paths = {
        "FixedReceptacle": f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle",
        "BodyAssembly": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly",
        "CouplingNut": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut",
    }
    try:
        return paths[owner]
    except KeyError as exc:
        raise ValueError(f"unknown rigid owner {owner}") from exc


def _domain_values(field: str, raw_domain: Any, labels: Sequence[str]) -> tuple[Any, ...]:
    if field == "label" and raw_domain == "exact_contact_layout_label_order_61":
        return tuple(labels)
    if isinstance(raw_domain, str):
        match = _RANGE_PREFIX.match(raw_domain)
        if match:
            low, high = (int(value) for value in match.groups())
            return tuple(range(low, high + 1))
    raise ValueError(f"unsupported frozen index domain for {field}: {raw_domain!r}")


def _expand_family_paths(
    family_name: str, family: Mapping[str, Any], labels: Sequence[str]
) -> tuple[str, ...]:
    domains = _mapping(family["index_domains"], f"{family_name}.index_domains")
    output: list[str] = []
    for template in family["path_templates"]:
        fields: list[str] = []
        for _, field_name, _, _ in Formatter().parse(template):
            if field_name and field_name not in fields:
                fields.append(field_name)
        values = [_domain_values(field, domains[field], labels) for field in fields]
        for combination in product(*values):
            output.append(template.format(**dict(zip(fields, combination))))
    if len(output) != family["expected_leaf_count_nominal"]:
        raise ValueError(f"trusted path expansion for {family_name} does not close")
    if len(set(output)) != len(output):
        raise ValueError(f"trusted path expansion for {family_name} contains duplicates")
    return tuple(output)


def _trusted_collider_inventory(
    model: PhysicalModelContract,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    blueprint = model.document["a2_collision_authoring_blueprint"]
    filtering = blueprint["filtering"]
    families = filtering["primitive_family_definitions"]
    leaf_contract = filtering["realized_leaf_readback_contract"]
    labels = tuple(row[0] for row in blueprint["contact_layout"]["positions_in_exactly"])
    inventory: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    overrides = leaf_contract["analytic_primitive_overrides"]
    for family_name in sorted(families):
        family = families[family_name]
        realized = {
            "typeName": leaf_contract["default_typeName"],
            "geometry_type": leaf_contract["default_geometry_type"],
            "physics_approximation": leaf_contract["default_physics_approximation"],
        }
        realized.update(overrides.get(family_name, {}))
        for path in _expand_family_paths(family_name, family, labels):
            identity = (family_name, path, leaf_contract["collider_index_per_leaf_prim"])
            inventory[identity] = {
                "family": family_name,
                "prim_path": path,
                "collider_index": leaf_contract["collider_index_per_leaf_prim"],
                "owner": _owner_path(family["owner"]),
                "materialRole": family["material_role"],
                "responseRole": family["response_role"],
                "collision_group": leaf_contract["collision_group_path_template"].format(
                    primitive_family=family_name
                ),
                **realized,
            }
    expected_count = model.document["solver_profile"]["resolved_readback_result_contract"][
        "expected_collider_row_count"
    ]
    if len(inventory) != expected_count:
        raise ValueError("trusted collider inventory count does not close")
    return inventory


def _trusted_property_inventory(
    model: PhysicalModelContract,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    joint = model.document["solver_profile"]["authored_attribute_contract"][
        "nut_body_D6_joint"
    ]
    prim_path = f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNutJoint"
    semantic_id = "nut_body_D6_joint"
    specifications: list[tuple[str, str, Any, bool, list[str]]] = [
        ("typeName", "__typeName", joint["typeName"], True, []),
        ("attribute", "physics:jointEnabled", joint["physics:jointEnabled"], True, []),
        ("attribute", "physics:collisionEnabled", joint["physics:collisionEnabled"], True, []),
        ("relationship", "physics:body0", joint["physics:body0"], True, [joint["physics:body0"]]),
        ("relationship", "physics:body1", joint["physics:body1"], True, [joint["physics:body1"]]),
        ("attribute", "physics:localPos0", joint["physics:localPos0"], True, []),
        ("attribute", "physics:localPos1", joint["physics:localPos1"], True, []),
        ("attribute", "physics:localRot0_wxyz", joint["physics:localRot0_wxyz"], True, []),
        ("attribute", "physics:localRot1_wxyz", joint["physics:localRot1_wxyz"], True, []),
        ("required_schemas", "__requiredAppliedSchemas", joint["required_applied_schemas"], True, []),
    ]
    for name, value in joint.items():
        if name.startswith("limit:"):
            specifications.append(("attribute", name, value, True, []))
    specifications.extend(
        [
            ("forbidden_schemas_absent", "__forbiddenAppliedSchemasAbsent", joint["forbidden_applied_schemas"], False, []),
            ("forbidden_properties_absent", "__rotZLimitOrDrivePropertiesAbsent", joint["rotZ_limit_or_drive_properties_must_be_absent"], False, []),
        ]
    )
    inventory: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for kind, property_name, expected_value, authored, targets in specifications:
        identity = (semantic_id, prim_path, property_name)
        inventory[identity] = {
            "semantic_id": semantic_id,
            "prim_path": prim_path,
            "readback_kind": kind,
            "property_name": property_name,
            "expected_value": expected_value,
            "hasAuthoredOpinion": authored,
            "relationship_targets": targets,
        }
    expected_count = model.document["solver_profile"]["resolved_readback_result_contract"][
        "expected_property_row_count"
    ]
    if len(inventory) != expected_count:
        raise ValueError(
            f"trusted D6 property inventory count is {len(inventory)}, expected {expected_count}"
        )
    return inventory


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _trusted_family_algebra(
    model: PhysicalModelContract,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[Any, ...], Mapping[str, Any]],
]:
    filtering = model.document["a2_collision_authoring_blueprint"]["filtering"]
    families = filtering["primitive_family_definitions"]
    primitive_names = set(families)
    composites = filtering["composite_family_definitions"]
    counts = {name: data["expected_leaf_count_nominal"] for name, data in families.items()}

    def resolve(name: str) -> set[str]:
        if name in primitive_names:
            return {name}
        definition = composites[name]
        included = (
            set(primitive_names)
            if definition.get("include_all_primitive_families")
            else set(definition.get("include", []))
        )
        return included - set(definition.get("exclude", []))

    all_pairs = {
        _canonical_pair(left, right)
        for left in primitive_names
        for right in primitive_names
    }
    declared = filtering["rule_expansion_contract"]["declared_cross_pairs"]
    matched: set[tuple[str, str]] = set()
    rule_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    default_rule: Mapping[str, Any] | None = None
    for rule in filtering["family_pair_rules"]:
        if rule["expansion"] == "unordered_complement_of_all_prior_explicit_rules":
            default_rule = rule
            continue
        if rule["expansion"] == "cartesian":
            pairs = {
                _canonical_pair(left, right)
                for left in resolve(rule["left"])
                for right in resolve(rule["right"])
            }
        elif rule["expansion"] == "declared_cross_pairs":
            pairs = {
                _canonical_pair(left, right)
                for left, right in declared[rule["rule_id"]]
            }
        else:
            raise ValueError(f"unsupported frozen family-pair expansion {rule['expansion']}")
        if matched & pairs:
            raise ValueError("trusted family-pair rules overlap")
        for pair in pairs:
            rule_by_pair[pair] = rule
        matched |= pairs
    if default_rule is None:
        raise ValueError("trusted family-pair algebra lacks its default rule")
    for pair in all_pairs - matched:
        rule_by_pair[pair] = default_rule
    if set(rule_by_pair) != all_pairs:
        raise ValueError("trusted family-pair algebra does not cover all base pairs")

    response_roles = model.document["material_roles"]["response_roles"]
    group_authoring = filtering["collision_group_authoring"]
    group_template = group_authoring["group_path_template"]
    pair_inventory: dict[tuple[str, str], Mapping[str, Any]] = {}
    filter_inventory: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for pair in sorted(all_pairs):
        left, right = pair
        rule = rule_by_pair[pair]
        left_count, right_count = counts[left], counts[right]
        concrete_count = (
            left_count * (left_count - 1) // 2
            if left == right
            else left_count * right_count
        )
        left_group = group_template.format(primitive_family=left)
        right_group = group_template.format(primitive_family=right)
        compliant_side_count = sum(
            response_roles[families[name]["response_role"]]["class"] == "compliant"
            for name in pair
        )
        owner_classes = {families[name]["owner"] for name in pair}
        joint_gate: bool | str = (
            True if owner_classes == {"BodyAssembly", "CouplingNut"} else "not_applicable"
        )
        filtered_sources = [left_group] if rule["final_decision"] == "filtered" else []
        pair_inventory[pair] = {
            "left_family": left,
            "right_family": right,
            "decision_rule_id": rule["rule_id"],
            "expected_decision": rule["final_decision"],
            "expected_response_class": rule["response"],
            "expected_left_member_count": left_count,
            "expected_right_member_count": right_count,
            "expected_concrete_leaf_pair_count": concrete_count,
            "compliant_side_count": compliant_side_count,
            "collision_group_sources": [left_group] if left == right else [left_group, right_group],
            "filteredPairs_sources": filtered_sources,
            "ancestor_filter_sources": [],
            "joint_collision_gate": joint_gate,
            "matched_final_rule_count": 1,
        }
        if rule["final_decision"] == "filtered":
            filter_identity = (
                left_group,
                group_authoring["group_schema"],
                group_authoring["filtered_group_relationship"],
                left,
                right,
            )
            filter_inventory[filter_identity] = {
                "source_prim_path": left_group,
                "source_schema": group_authoring["group_schema"],
                "source_property_or_relationship": group_authoring["filtered_group_relationship"],
                "affected_left_family": left,
                "affected_right_family": right,
                "decision_rule_id": rule["rule_id"],
                "expected_effect": "filters",
                "expected_concrete_leaf_pair_count": concrete_count,
            }
    result_contract = model.document["solver_profile"]["resolved_readback_result_contract"]
    if len(pair_inventory) != result_contract["expected_family_pair_row_count"]:
        raise ValueError("trusted family-pair row count does not close")
    if len(filter_inventory) != result_contract["expected_filter_source_row_count"]:
        raise ValueError("trusted filter-source row count does not close")
    return pair_inventory, filter_inventory


def _finite_bounds(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    if any(not isinstance(point, list) or len(point) != 3 for point in value):
        return False
    try:
        low = [float(number) for number in value[0]]
        high = [float(number) for number in value[1]]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(number) for number in low + high) and all(
        lower < upper for lower, upper in zip(low, high)
    )


def _topology_signature_ok(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip()
    lowered = normalized.lower()
    return not any(part in lowered for part in FORBIDDEN_METADATA_KEY_PARTS) and not bool(
        _HEX_FINGERPRINT.fullmatch(normalized)
    )


def _resolved_value_matches(actual: Any, expected: Any) -> bool:
    """Compare authored schema values after their native USD storage conversion."""

    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, Real) and isinstance(actual, Real):
        return math.isclose(
            float(actual), float(expected), rel_tol=2.0e-7, abs_tol=2.0e-12
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            _resolved_value_matches(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _collider_row_pass(
    row: Mapping[str, Any], expected: Mapping[str, Any], model: PhysicalModelContract
) -> bool:
    response = model.document["material_roles"]["response_roles"][expected["responseRole"]]
    if response["class"] == "hard":
        expected_stiffness = response["compliant_stiffness_n_m"]
        expected_damping = response["compliant_damping_n_s_m"]
    else:
        expected_stiffness = response["nominal_stiffness_n_m"]
        expected_damping = response["nominal_damping_n_s_m"]
    return (
        all(row[field] == expected[field] for field in ("family", "prim_path", "collider_index"))
        and row["typeName"] == expected["typeName"]
        and isinstance(row["appliedSchemas"], list)
        and "PhysicsCollisionAPI" in row["appliedSchemas"]
        and row["collisionEnabled"] is True
        and _finite_bounds(row["local_bounds"])
        and _finite_bounds(row["world_bounds_at_canonical_pose"])
        and row["geometry_type"] == expected["geometry_type"]
        and row["physics_approximation"] == expected["physics_approximation"]
        and row["closed_manifold"] is True
        and row["positive_volume"] is True
        and row["convex"] is True
        and _topology_signature_ok(row["topology_signature"])
        and row["materialRole"] == expected["materialRole"]
        and row["responseRole"] == expected["responseRole"]
        and isinstance(row["effective_physics_material_binding"], str)
        and bool(row["effective_physics_material_binding"])
        and isinstance(row["material_binding_source_prim"], str)
        and bool(row["material_binding_source_prim"])
        and _resolved_value_matches(
            row["resolved_compliant_stiffness_n_m"], expected_stiffness
        )
        and _resolved_value_matches(
            row["resolved_compliant_damping_n_s_m"], expected_damping
        )
        and row["resolved_accelerationSpring"] is False
        and row["nearest_rigid_body_owner"] == expected["owner"]
        and row["owner_rigidBodyEnabled"] is True
        and row["owner_kinematicEnabled"] is False
        and row["offset_class"] == "fine_connector"
        and _resolved_value_matches(row["contactOffset_m"], 0.00001)
        and _resolved_value_matches(row["restOffset_m"], 0.0)
        and row["collision_group_memberships"] == [expected["collision_group"]]
        and row["filteredPairs_sources"] == []
    )


def _property_row_pass(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    static_fields = (
        "semantic_id", "prim_path", "readback_kind", "property_name",
        "expected_value", "hasAuthoredOpinion", "relationship_targets",
    )
    return all(row[field] == expected[field] for field in static_fields) and (
        _resolved_value_matches(row["resolved_value"], expected["expected_value"])
    )


def _family_pair_row_pass(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        all(row[field] == expected[field] for field in expected)
        and row["resolved_decision"] == expected["expected_decision"]
        and row["resolved_left_member_count"] == expected["expected_left_member_count"]
        and row["resolved_right_member_count"] == expected["expected_right_member_count"]
        and row["resolved_concrete_leaf_pair_count"] == expected["expected_concrete_leaf_pair_count"]
        and (expected["expected_response_class"] != "compliant" or expected["compliant_side_count"] == 1)
        and not (expected["expected_decision"] == "enabled" and expected["filteredPairs_sources"])
    )


def _filter_row_pass(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        all(row[field] == expected[field] for field in expected)
        and row["resolved_effect"] == expected["expected_effect"]
        and row["resolved_concrete_leaf_pair_count"] == expected["expected_concrete_leaf_pair_count"]
    )


@dataclass(frozen=True)
class A2ReadbackValidation:
    asset_path: Path
    collider_row_count: int
    property_row_count: int
    family_pair_row_count: int
    filter_source_row_count: int
    recomputed_summary: Mapping[str, int]
    release_evidence: bool


def validate_a2_resolved_readback_result(
    actual_result: Mapping[str, Any] | Path | str,
    *,
    model: PhysicalModelContract | None = None,
) -> A2ReadbackValidation:
    """Validate a candidate artifact using only internally derived expectations."""

    active_model = model or load_physical_model_contract()
    solver = active_model.document["solver_profile"]
    contract = solver["resolved_readback_result_contract"]
    actual = _load_candidate(actual_result)
    _reject_fingerprint_metadata(actual)
    if set(actual) != set(contract["required_top_level_fields"]):
        raise ValueError("candidate top-level field inventory changed")
    if actual["schema_version"] != contract["schema_version"]:
        raise ValueError(
            f"candidate schema_version must be {contract['schema_version']}"
        )
    if actual["generator_id"] != contract["generator_id"]:
        raise ValueError("candidate generator identity changed")
    if actual["contract_revision"] != contract["contract_revision"]:
        raise ValueError("candidate contract revision changed")
    if actual["root_prim"] != SUCCESSOR_ROOT_PRIM:
        raise ValueError("candidate root prim changed")
    authorized_asset = _authorized_asset_path(active_model)
    if Path(str(actual["asset_path"])).expanduser().resolve() != authorized_asset:
        raise ValueError("candidate asset path is not the authorized A2 output")

    field_sets = {
        "collider_rows": set(solver["resolved_readback_required_fields"]),
        "property_rows": set(solver["resolved_property_readback_required_fields"]),
        "family_pair_rows": set(solver["resolved_family_pair_readback_required_fields"]),
        "filter_source_rows": set(solver["resolved_filter_source_row_required_fields"]),
    }
    rows = {name: _rows(actual[name], f"candidate.{name}") for name in COLLECTION_NAMES}
    for collection, collection_rows in rows.items():
        for index, row in enumerate(collection_rows):
            _require_exact_keys(row, field_sets[collection], f"candidate.{collection}[{index}]")

    expected_colliders = _trusted_collider_inventory(active_model)
    expected_properties = _trusted_property_inventory(active_model)
    expected_pairs, expected_filters = _trusted_family_algebra(active_model)
    expected_by_collection = {
        "collider_rows": expected_colliders,
        "property_rows": expected_properties,
        "family_pair_rows": expected_pairs,
        "filter_source_rows": expected_filters,
    }
    identity_fields = {
        "collider_rows": contract["collider_row_identity_fields"],
        "property_rows": contract["property_row_identity_fields"],
        "family_pair_rows": contract["family_pair_row_identity_fields"],
        "filter_source_rows": contract["filter_source_row_identity_fields"],
    }
    indexed: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    duplicates: dict[str, int] = {}
    for collection in COLLECTION_NAMES:
        indexed[collection], duplicates[collection] = _index_rows(
            rows[collection], identity_fields[collection]
        )

    pass_functions = {
        "collider_rows": lambda row, expected: _collider_row_pass(row, expected, active_model),
        "property_rows": _property_row_pass,
        "family_pair_rows": _family_pair_row_pass,
        "filter_source_rows": _filter_row_pass,
    }
    failures: dict[str, int] = {}
    for collection in COLLECTION_NAMES:
        failures[collection] = 0
        common = indexed[collection].keys() & expected_by_collection[collection].keys()
        for identity in common:
            row = indexed[collection][identity]
            passed = pass_functions[collection](row, expected_by_collection[collection][identity])
            if row["pass"] is not passed:
                raise ValueError(f"candidate {collection} row has an untrusted pass claim")
            failures[collection] += int(not passed)

    def missing(collection: str) -> int:
        return len(expected_by_collection[collection].keys() - indexed[collection].keys())

    def unexpected(collection: str) -> int:
        return len(indexed[collection].keys() - expected_by_collection[collection].keys())

    actual_collider_paths = {row["prim_path"] for row in rows["collider_rows"]}
    expected_collider_paths = {value["prim_path"] for value in expected_colliders.values()}
    expected_family_counts = Counter(value["family"] for value in expected_colliders.values())
    actual_family_counts = Counter(row["family"] for row in rows["collider_rows"])
    common_collider_ids = indexed["collider_rows"].keys() & expected_colliders.keys()
    common_pair_ids = indexed["family_pair_rows"].keys() & expected_pairs.keys()
    expected_filtered_pair_ids = {
        pair for pair, value in expected_pairs.items() if value["expected_decision"] == "filtered"
    }
    actual_filter_pair_ids = {
        (row["affected_left_family"], row["affected_right_family"])
        for row in rows["filter_source_rows"]
    }
    recomputed = {
        "failed_collider_row_count": failures["collider_rows"],
        "failed_property_row_count": failures["property_rows"],
        "failed_family_pair_row_count": failures["family_pair_rows"],
        "failed_filter_source_row_count": failures["filter_source_rows"],
        "missing_expected_collider_row_count": missing("collider_rows"),
        "unexpected_collider_row_count": unexpected("collider_rows"),
        "duplicate_collider_row_identity_count": duplicates["collider_rows"],
        "missing_expected_property_row_count": missing("property_rows"),
        "unexpected_property_row_count": unexpected("property_rows"),
        "duplicate_property_row_identity_count": duplicates["property_rows"],
        "missing_expected_family_pair_row_count": missing("family_pair_rows"),
        "unexpected_family_pair_row_count": unexpected("family_pair_rows"),
        "duplicate_family_pair_row_identity_count": duplicates["family_pair_rows"],
        "missing_expected_filter_source_row_count": missing("filter_source_rows"),
        "unexpected_filter_source_row_count": unexpected("filter_source_rows"),
        "duplicate_filter_source_row_identity_count": duplicates["filter_source_rows"],
        "missing_semantic_prim_count": len(expected_collider_paths - actual_collider_paths),
        "unexpected_semantic_prim_count": len(actual_collider_paths - expected_collider_paths),
        "unclassified_connector_collider_count": sum(
            row["family"] not in expected_family_counts for row in rows["collider_rows"]
        ),
        "family_member_count_mismatch_count": sum(
            actual_family_counts[name] != count for name, count in expected_family_counts.items()
        ),
        "unmatched_base_family_pair_count": missing("family_pair_rows"),
        "multiply_matched_base_family_pair_count": duplicates["family_pair_rows"],
        "concrete_leaf_pair_count_mismatch_count": sum(
            indexed["family_pair_rows"][identity]["resolved_concrete_leaf_pair_count"]
            != expected_pairs[identity]["expected_concrete_leaf_pair_count"]
            for identity in common_pair_ids
        ),
        "unassigned_material_role_collider_count": sum(
            not bool(row["materialRole"]) for row in rows["collider_rows"]
        ),
        "unassigned_response_role_collider_count": sum(
            not bool(row["responseRole"]) for row in rows["collider_rows"]
        ),
        "intended_enabled_family_pair_filtered_count": sum(
            expected_pairs[identity]["expected_decision"] == "enabled"
            and indexed["family_pair_rows"][identity]["resolved_decision"] != "enabled"
            for identity in common_pair_ids
        ),
        "filter_source_coverage_error_count": len(expected_filtered_pair_ids ^ actual_filter_pair_ids),
        "rigid_owner_resolution_error_count": sum(
            indexed["collider_rows"][identity]["nearest_rigid_body_owner"]
            != expected_colliders[identity]["owner"]
            for identity in common_collider_ids
        ),
        "collision_group_membership_error_count": sum(
            indexed["collider_rows"][identity]["collision_group_memberships"]
            != [expected_colliders[identity]["collision_group"]]
            for identity in common_collider_ids
        ),
        "geometry_metadata_disagreement_count": sum(
            indexed["collider_rows"][identity]["typeName"] != expected_colliders[identity]["typeName"]
            or indexed["collider_rows"][identity]["geometry_type"] != expected_colliders[identity]["geometry_type"]
            or not _finite_bounds(indexed["collider_rows"][identity]["local_bounds"])
            or not _finite_bounds(indexed["collider_rows"][identity]["world_bounds_at_canonical_pose"])
            or not _topology_signature_ok(indexed["collider_rows"][identity]["topology_signature"])
            for identity in common_collider_ids
        ),
        "nonconvex_dynamic_collider_count": sum(
            row["owner_rigidBodyEnabled"] is True
            and row["owner_kinematicEnabled"] is False
            and row["convex"] is not True
            for row in rows["collider_rows"]
        ),
        "automatic_collision_approximation_count": sum(
            indexed["collider_rows"][identity]["physics_approximation"]
            != expected_colliders[identity]["physics_approximation"]
            for identity in common_collider_ids
        ),
        "ancestor_filter_covering_intended_pair_count": sum(
            expected_pairs[identity]["expected_decision"] == "enabled"
            and bool(indexed["family_pair_rows"][identity]["ancestor_filter_sources"])
            for identity in common_pair_ids
        ),
        "compliant_pair_with_zero_or_two_compliant_sides_count": sum(
            expected_pairs[identity]["expected_response_class"] == "compliant"
            and indexed["family_pair_rows"][identity]["compliant_side_count"] != 1
            for identity in common_pair_ids
        ),
    }
    required_summary = dict(contract["required_summary_counts"])
    if set(recomputed) != set(required_summary):
        raise ValueError("candidate summary implementation is out of contract")
    if dict(_mapping(actual["summary"], "candidate.summary")) != recomputed:
        raise ValueError("claimed A2 summary differs from internal recomputation")
    nonzero = {name: value for name, value in recomputed.items() if value != 0}
    if nonzero:
        raise ValueError(f"A2 reader candidate failed: {nonzero}")
    return A2ReadbackValidation(
        asset_path=authorized_asset,
        collider_row_count=len(rows["collider_rows"]),
        property_row_count=len(rows["property_rows"]),
        family_pair_row_count=len(rows["family_pair_rows"]),
        filter_source_row_count=len(rows["filter_source_rows"]),
        recomputed_summary=recomputed,
        release_evidence=False,
    )


def _bbox_as_lists(box: Any) -> list[list[float]]:
    minimum = box.GetMin()
    maximum = box.GetMax()
    return [
        [float(minimum[index]) for index in range(3)],
        [float(maximum[index]) for index in range(3)],
    ]


def _canonical_world_bounds(
    owner_local_bounds: list[list[float]], owner_path: str, model: PhysicalModelContract
) -> list[list[float]]:
    pose = model.document["a2_collision_authoring_blueprint"]["global"][
        "canonical_readback_pose"
    ]
    if owner_path.endswith("/FixedReceptacle"):
        translation = pose["fixed_receptacle_translation_m"]
        transform = lambda x, y, z: (x, y, z)
    else:
        translation = pose["loose_plug_translation_m"]
        # The frozen qX180 mapping is [x, -y, -z].
        transform = lambda x, y, z: (x, -y, -z)
    corners = []
    for x_value in owner_local_bounds[0][0], owner_local_bounds[1][0]:
        for y_value in owner_local_bounds[0][1], owner_local_bounds[1][1]:
            for z_value in owner_local_bounds[0][2], owner_local_bounds[1][2]:
                tx, ty, tz = transform(x_value, y_value, z_value)
                corners.append(
                    (
                        tx + float(translation[0]),
                        ty + float(translation[1]),
                        tz + float(translation[2]),
                    )
                )
    return [
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
    ]


def _mesh_geometry_readback(mesh: Any, tolerance_m: float) -> Mapping[str, Any]:
    points = [tuple(float(value) for value in point) for point in mesh.GetPointsAttr().Get()]
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
    flat = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
    if not points or not counts or sum(counts) != len(flat):
        return {
            "closed_manifold": False,
            "positive_volume": False,
            "convex": False,
            "topology_signature": "type=Mesh;invalid=empty_or_inconsistent_topology",
        }
    faces: list[list[int]] = []
    edge_counts: Counter[tuple[int, int]] = Counter()
    offset = 0
    for count in counts:
        face = flat[offset : offset + count]
        offset += count
        if count < 3 or any(index < 0 or index >= len(points) for index in face):
            return {
                "closed_manifold": False,
                "positive_volume": False,
                "convex": False,
                "topology_signature": "type=Mesh;invalid=face_index_or_cardinality",
            }
        faces.append(face)
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edge_counts[tuple(sorted((first, second)))] += 1
    closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    signed_volume = 0.0
    convex = True
    maximum_plane_violation = 0.0
    for face in faces:
        p0 = points[face[0]]
        for index in range(1, len(face) - 1):
            p1, p2 = points[face[index]], points[face[index + 1]]
            cross = (
                p1[1] * p2[2] - p1[2] * p2[1],
                p1[2] * p2[0] - p1[0] * p2[2],
                p1[0] * p2[1] - p1[1] * p2[0],
            )
            signed_volume += sum(p0[axis] * cross[axis] for axis in range(3)) / 6.0
        p1, p2 = points[face[1]], points[face[2]]
        first_edge = tuple(p1[axis] - p0[axis] for axis in range(3))
        second_edge = tuple(p2[axis] - p0[axis] for axis in range(3))
        normal = (
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
        )
        norm = math.sqrt(sum(value * value for value in normal))
        if norm <= 0.0:
            convex = False
            continue
        for point in points:
            distance = sum(
                normal[axis] * (point[axis] - p0[axis]) for axis in range(3)
            ) / norm
            maximum_plane_violation = max(maximum_plane_violation, distance)
    convex = convex and maximum_plane_violation <= tolerance_m
    return {
        "closed_manifold": closed,
        "positive_volume": signed_volume > 0.0,
        "convex": convex,
        "topology_signature": (
            f"type=Mesh;points={len(points)};faces={len(faces)};"
            f"edges={len(edge_counts)};closed={str(closed).lower()};"
            f"convex={str(convex).lower()}"
        ),
    }


def _validate_cooking_representation(
    prim: Any,
    *,
    model: PhysicalModelContract,
    PhysxSchema: Any,
    UsdGeom: Any,
) -> None:
    contract = model.document["convex_cooking_representation"]
    representation = prim.GetAttribute("kcg:cookingRepresentation")
    value = representation.Get() if representation else None
    if prim.IsA(UsdGeom.Mesh):
        if value != contract["representation_id"]:
            raise ValueError(f"mesh cooking representation changed at {prim.GetPath()}")
        if not prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI):
            raise ValueError(f"missing PhysxConvexHullCollisionAPI at {prim.GetPath()}")
        convex = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        minimum = convex.GetMinThicknessAttr()
        expected_minimum = contract[
            "physxConvexHullCollision:minThickness_local_units"
        ]
        if (
            not minimum
            or not minimum.HasAuthoredValueOpinion()
            or not _resolved_value_matches(minimum.Get(), expected_minimum)
        ):
            raise ValueError(f"convex minimum thickness changed at {prim.GetPath()}")
        xformable = UsdGeom.Xformable(prim)
        operations = xformable.GetOrderedXformOps()
        if xformable.GetResetXformStack() or len(operations) != 1:
            raise ValueError(f"mesh cooking transform stack changed at {prim.GetPath()}")
        operation = operations[0]
        if (
            operation.GetOpName() != "xformOp:scale"
            or operation.GetOpType() != UsdGeom.XformOp.TypeScale
        ):
            raise ValueError(f"mesh cooking scale operation changed at {prim.GetPath()}")
        actual_scale = [float(item) for item in operation.Get()]
        if not _resolved_value_matches(
            actual_scale, contract["mesh_uniform_scale_xyz"]
        ):
            raise ValueError(f"mesh cooking scale changed at {prim.GetPath()}")
    else:
        if value != "analytic_not_applicable":
            raise ValueError(f"analytic cooking exception changed at {prim.GetPath()}")
        if prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI):
            raise ValueError(f"analytic collider has convex cooking API at {prim.GetPath()}")


def _nearest_rigid_owner(prim: Any, UsdPhysics: Any) -> Any:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _direct_physics_material(prim: Any, stage: Any) -> tuple[str, str]:
    relationship = prim.GetRelationship("material:binding:physics")
    if not relationship or not relationship.HasAuthoredTargets():
        return "", ""
    targets = relationship.GetTargets()
    if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]).IsValid():
        return "", ""
    return str(targets[0]), str(prim.GetPath())


def _quat_wxyz(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ]


def _property_resolved_value(prim: Any, expected: Mapping[str, Any]) -> tuple[Any, bool, list[str]]:
    property_name = str(expected["property_name"])
    kind = str(expected["readback_kind"])
    if property_name == "__typeName":
        return str(prim.GetTypeName()), bool(prim.GetTypeName()), []
    if property_name == "__requiredAppliedSchemas":
        actual = list(prim.GetAppliedSchemas())
        required = list(expected["expected_value"])
        resolved = required if all(name in actual for name in required) else actual
        return resolved, bool(prim.GetMetadata("apiSchemas")), []
    if property_name == "__forbiddenAppliedSchemasAbsent":
        actual = list(prim.GetAppliedSchemas())
        forbidden = list(expected["expected_value"])
        present = [name for name in forbidden if name in actual]
        return (forbidden if not present else present), False, []
    if property_name == "__rotZLimitOrDrivePropertiesAbsent":
        names = [str(prop.GetName()) for prop in prim.GetProperties()]
        absent = not any(
            name.startswith("limit:rotZ:") or name.startswith("drive:rotZ:")
            for name in names
        )
        return absent, False, []
    if kind == "relationship":
        relationship = prim.GetRelationship(property_name)
        targets = [str(path) for path in relationship.GetTargets()] if relationship else []
        resolved: Any = targets[0] if len(targets) == 1 else targets
        return resolved, bool(relationship and relationship.HasAuthoredTargets()), targets
    authored_name = property_name
    if property_name == "physics:localRot0_wxyz":
        authored_name = "physics:localRot0"
    elif property_name == "physics:localRot1_wxyz":
        authored_name = "physics:localRot1"
    attribute = prim.GetAttribute(authored_name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None, False, []
    value = attribute.Get()
    if property_name.endswith("_wxyz"):
        value = _quat_wxyz(value)
    elif hasattr(value, "__len__") and not isinstance(value, str):
        try:
            value = [float(item) for item in value]
        except TypeError:
            pass
    elif isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
    return value, True, []


def _read_composed_stage_result(
    asset_path: Path, model: PhysicalModelContract
) -> Mapping[str, Any]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError("authorized A2 asset could not be opened")
    root = stage.GetPrimAtPath(SUCCESSOR_ROOT_PRIM)
    if not root.IsValid():
        raise ValueError("authorized A2 asset lacks its frozen root prim")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("authorized A2 stage metersPerUnit changed")
    if UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
        raise ValueError("authorized A2 stage kilogramsPerUnit changed")

    expected_colliders = _trusted_collider_inventory(model)
    expected_properties = _trusted_property_inventory(model)
    expected_pairs, expected_filters = _trusted_family_algebra(model)
    groups = {
        str(prim.GetPath()): UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.CollisionGroup)
    }
    group_members: dict[str, set[str]] = {}
    member_groups: dict[str, list[str]] = defaultdict(list)
    group_filters: dict[str, set[str]] = {}
    for path, group in groups.items():
        collection = group.GetCollidersCollectionAPI()
        members = {str(target) for target in collection.GetIncludesRel().GetTargets()}
        group_members[path] = members
        for member in members:
            member_groups[member].append(path)
        group_filters[path] = {
            str(target) for target in group.GetFilteredGroupsRel().GetTargets()
        }

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    tolerance = float(
        model.document["a2_collision_authoring_blueprint"]["global"][
            "geometry_abs_tolerance_m"
        ]
    )
    collider_rows = []
    for identity in sorted(expected_colliders):
        expected = expected_colliders[identity]
        prim = stage.GetPrimAtPath(expected["prim_path"])
        if not prim.IsValid():
            continue
        _validate_cooking_representation(
            prim,
            model=model,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
        )
        owner = _nearest_rigid_owner(prim, UsdPhysics)
        owner_path = str(owner.GetPath()) if owner is not None else ""
        if owner is not None:
            local_box = bbox_cache.ComputeRelativeBound(prim, owner).ComputeAlignedBox()
            local_bounds = _bbox_as_lists(local_box)
            world_bounds = _canonical_world_bounds(local_bounds, owner_path, model)
            rigid = UsdPhysics.RigidBodyAPI(owner)
            owner_enabled = rigid.GetRigidBodyEnabledAttr().Get()
            owner_kinematic = rigid.GetKinematicEnabledAttr().Get()
        else:
            local_bounds = []
            world_bounds = []
            owner_enabled = None
            owner_kinematic = None
        if prim.IsA(UsdGeom.Mesh):
            cooking_scale = float(
                model.document["convex_cooking_representation"][
                    "mesh_uniform_scale_xyz"
                ][0]
            )
            geometry = _mesh_geometry_readback(
                UsdGeom.Mesh(prim), tolerance / cooking_scale
            )
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            geometry_type = "explicit_convex_mesh"
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            radius = cylinder.GetRadiusAttr().Get()
            height = cylinder.GetHeightAttr().Get()
            valid = bool(radius and height and radius > 0.0 and height > 0.0)
            geometry = {
                "closed_manifold": valid,
                "positive_volume": valid,
                "convex": valid,
                "topology_signature": "type=Cylinder;axis=Z;closed=true;convex=true",
            }
            approximation = "none"
            geometry_type = "analytic_cylinder"
        elif prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            radius = sphere.GetRadiusAttr().Get()
            valid = bool(radius and radius > 0.0)
            geometry = {
                "closed_manifold": valid,
                "positive_volume": valid,
                "convex": valid,
                "topology_signature": "type=Sphere;closed=true;convex=true",
            }
            approximation = "none"
            geometry_type = "analytic_sphere"
        elif prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            radius = capsule.GetRadiusAttr().Get()
            height = capsule.GetHeightAttr().Get()
            valid = bool(radius and height and radius > 0.0 and height > 0.0)
            geometry = {
                "closed_manifold": valid,
                "positive_volume": valid,
                "convex": valid,
                "topology_signature": "type=Capsule;closed=true;convex=true",
            }
            approximation = "none"
            geometry_type = "analytic_capsule"
        else:
            geometry = {
                "closed_manifold": False,
                "positive_volume": False,
                "convex": False,
                "topology_signature": "type=unexpected",
            }
            approximation = ""
            geometry_type = "unexpected"
        material_path, binding_source = _direct_physics_material(prim, stage)
        material_prim = stage.GetPrimAtPath(material_path) if material_path else None
        stiffness = (
            material_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get()
            if material_prim is not None
            else None
        )
        damping = (
            material_prim.GetAttribute("physxMaterial:compliantContactDamping").Get()
            if material_prim is not None
            else None
        )
        acceleration = (
            material_prim.GetAttribute(
                "physxMaterial:compliantContactAccelerationSpring"
            ).Get()
            if material_prim is not None
            else None
        )
        row = {
            "family": identity[0],
            "prim_path": identity[1],
            "collider_index": identity[2],
            "typeName": str(prim.GetTypeName()),
            "appliedSchemas": list(prim.GetAppliedSchemas()),
            "collisionEnabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            "local_bounds": local_bounds,
            "world_bounds_at_canonical_pose": world_bounds,
            "geometry_type": geometry_type,
            "physics_approximation": str(approximation),
            **geometry,
            "materialRole": prim.GetAttribute("kcg:materialRole").Get(),
            "responseRole": prim.GetAttribute("kcg:responseRole").Get(),
            "effective_physics_material_binding": material_path,
            "material_binding_source_prim": binding_source,
            "resolved_compliant_stiffness_n_m": stiffness,
            "resolved_compliant_damping_n_s_m": damping,
            "resolved_accelerationSpring": acceleration,
            "nearest_rigid_body_owner": owner_path,
            "owner_rigidBodyEnabled": owner_enabled,
            "owner_kinematicEnabled": owner_kinematic,
            "offset_class": prim.GetAttribute("kcg:offsetClass").Get(),
            "contactOffset_m": prim.GetAttribute("physxCollision:contactOffset").Get(),
            "restOffset_m": prim.GetAttribute("physxCollision:restOffset").Get(),
            "collision_group_memberships": sorted(member_groups.get(identity[1], [])),
            "filteredPairs_sources": [],
        }
        row["pass"] = _collider_row_pass(row, expected, model)
        collider_rows.append(row)

    property_rows = []
    for identity in sorted(expected_properties):
        expected = expected_properties[identity]
        prim = stage.GetPrimAtPath(expected["prim_path"])
        if prim.IsValid():
            resolved, authored, targets = _property_resolved_value(prim, expected)
        else:
            resolved, authored, targets = None, False, []
        row = {
            **expected,
            "resolved_value": resolved,
            "hasAuthoredOpinion": authored,
            "relationship_targets": targets,
        }
        row["pass"] = _property_row_pass(row, expected)
        property_rows.append(row)

    family_pair_rows = []
    response_roles = model.document["material_roles"]["response_roles"]
    families = model.document["a2_collision_authoring_blueprint"]["filtering"][
        "primitive_family_definitions"
    ]
    joint = stage.GetPrimAtPath(f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNutJoint")
    joint_collision = joint.GetAttribute("physics:collisionEnabled").Get()
    for identity in sorted(expected_pairs):
        expected = expected_pairs[identity]
        left, right = identity
        left_group, right_group = expected["collision_group_sources"][:1][0], (
            expected["collision_group_sources"][-1]
        )
        filter_sources = []
        if right_group in group_filters.get(left_group, set()):
            filter_sources.append(left_group)
        if left_group != right_group and left_group in group_filters.get(right_group, set()):
            filter_sources.append(right_group)
        left_count = len(group_members.get(left_group, set()))
        right_count = len(group_members.get(right_group, set()))
        concrete_count = (
            left_count * (left_count - 1) // 2
            if left == right
            else left_count * right_count
        )
        compliant_count = sum(
            response_roles[families[name]["response_role"]]["class"] == "compliant"
            for name in identity
        )
        row = {
            **expected,
            "resolved_decision": "filtered" if filter_sources else "enabled",
            "resolved_left_member_count": left_count,
            "resolved_right_member_count": right_count,
            "resolved_concrete_leaf_pair_count": concrete_count,
            "compliant_side_count": compliant_count,
            "collision_group_sources": expected["collision_group_sources"],
            "filteredPairs_sources": sorted(filter_sources),
            "ancestor_filter_sources": [],
            "joint_collision_gate": (
                bool(joint_collision)
                if expected["joint_collision_gate"] is True
                else "not_applicable"
            ),
            "matched_final_rule_count": 1,
        }
        row["pass"] = _family_pair_row_pass(row, expected)
        family_pair_rows.append(row)

    filter_rows = []
    for identity in sorted(expected_filters):
        expected = expected_filters[identity]
        source = expected["source_prim_path"]
        right_group = (
            model.document["a2_collision_authoring_blueprint"]["filtering"][
                "collision_group_authoring"
            ]["group_path_template"].format(
                primitive_family=expected["affected_right_family"]
            )
        )
        effect = right_group in group_filters.get(source, set())
        left_count = len(group_members.get(source, set()))
        right_count = len(group_members.get(right_group, set()))
        concrete_count = (
            left_count * (left_count - 1) // 2
            if expected["affected_left_family"] == expected["affected_right_family"]
            else left_count * right_count
        )
        row = {
            **expected,
            "resolved_effect": "filters" if effect else "does_not_filter",
            "resolved_concrete_leaf_pair_count": concrete_count,
        }
        row["pass"] = _filter_row_pass(row, expected)
        filter_rows.append(row)

    contract = model.document["solver_profile"]["resolved_readback_result_contract"]
    return {
        "schema_version": contract["schema_version"],
        "generator_id": contract["generator_id"],
        "contract_revision": contract["contract_revision"],
        "asset_path": str(asset_path),
        "root_prim": SUCCESSOR_ROOT_PRIM,
        "collider_rows": collider_rows,
        "property_rows": property_rows,
        "family_pair_rows": family_pair_rows,
        "filter_source_rows": filter_rows,
        "summary": dict(contract["required_summary_counts"]),
    }


def validate_a2_composed_asset_release(
    asset_path: Path | str,
    *,
    model: PhysicalModelContract | None = None,
) -> A2ReadbackValidation:
    """Read the authorized composed asset internally or fail closed."""

    active_model = model or load_physical_model_contract()
    authorized = _authorized_asset_path(active_model)
    requested = Path(asset_path).expanduser().resolve()
    if requested != authorized:
        raise ValueError("A2 release reader accepts only the authorized successor asset")
    if not active_model.a2_asset_authoring_allowed:
        raise PermissionError("A2 release is blocked while the A0 source freeze is open")
    if not requested.is_file():
        raise FileNotFoundError(f"authorized A2 asset does not exist: {requested}")
    candidate = _read_composed_stage_result(requested, active_model)
    validated = validate_a2_resolved_readback_result(
        candidate, model=active_model
    )
    return A2ReadbackValidation(
        asset_path=validated.asset_path,
        collider_row_count=validated.collider_row_count,
        property_row_count=validated.property_row_count,
        family_pair_row_count=validated.family_pair_row_count,
        filter_source_row_count=validated.filter_source_row_count,
        recomputed_summary=validated.recomputed_summary,
        release_evidence=True,
    )


__all__ = [
    "A2ReadbackValidation",
    "CONTRACT_REVISION",
    "GENERATOR_ID",
    "SCHEMA_VERSION",
    "validate_a2_composed_asset_release",
    "validate_a2_resolved_readback_result",
]
