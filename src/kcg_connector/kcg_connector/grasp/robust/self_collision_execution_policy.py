"""Hash-bound self-collision policy after structural-interface proof.

The base inventory deliberately restarts every unordered collision-link pair
as forbidden.  A route checker may stop testing one pair only when the same
verified URDF tree proves that the two collision-bearing links are the direct
parent and child of one joint.  SRDF rows and caller-supplied allowlists never
participate in this derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    AggregateRobotKinematicBinding,
)
from kcg_connector.grasp.robust.collision_contract import (
    SelfCollisionPairInventory,
)


METHOD_ID = "CARTS_HASH_BOUND_DIRECT_URDF_STRUCTURAL_INTERFACE_POLICY_V1"
STRUCTURAL_INTERFACE_RULE = (
    "EXCLUDE_ONLY_COLLISION_LINKS_THAT_ARE_DIRECT_PARENT_AND_CHILD_"
    "OF_ONE_HASH_BOUND_URDF_JOINT"
)
EXPECTED_LINK_COUNT = 17
EXPECTED_BASE_PAIR_COUNT = 136
EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT = 15
EXPECTED_FORBIDDEN_PAIR_COUNT = 121


class SelfCollisionExecutionPolicyError(ValueError):
    """Raised when structural-interface evidence is absent or inconsistent."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("self-collision policy error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if not first or not second or first == second:
        raise SelfCollisionExecutionPolicyError(
            "INVALID_LINK_PAIR", f"{first!r}:{second!r}"
        )
    return (first, second) if first < second else (second, first)


def _document(source: object) -> dict[str, object]:
    def field(name: str) -> object:
        return source[name] if isinstance(source, Mapping) else getattr(source, name)

    return {
        "method_id": field("method_id"),
        "structural_interface_rule": field("structural_interface_rule"),
        "aggregate_robot_model_sha256": field("aggregate_robot_model_sha256"),
        "base_self_pair_inventory_sha256": field(
            "base_self_pair_inventory_sha256"
        ),
        "link_names": list(field("link_names")),
        "all_pairs": [list(row) for row in field("all_pairs")],
        "structural_interface_pairs": [
            list(row) for row in field("structural_interface_pairs")
        ],
        "forbidden_pairs": [list(row) for row in field("forbidden_pairs")],
        "all_pair_count": field("all_pair_count"),
        "structural_interface_pair_count": field(
            "structural_interface_pair_count"
        ),
        "forbidden_pair_count": field("forbidden_pair_count"),
        "every_structural_interface_is_direct_urdf_parent_child": field(
            "every_structural_interface_is_direct_urdf_parent_child"
        ),
        "all_nonstructural_pairs_remain_forbidden": field(
            "all_nonstructural_pairs_remain_forbidden"
        ),
        "srdf_exemptions_applied": field("srdf_exemptions_applied"),
        "manual_pair_allowlist_used": field("manual_pair_allowlist_used"),
        "online_truth_used": field("online_truth_used"),
    }


def _certificate_sha256(source: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _document(source),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SelfCollisionExecutionPolicyCertificate:
    method_id: str
    structural_interface_rule: str
    aggregate_robot_model_sha256: str
    base_self_pair_inventory_sha256: str
    link_names: tuple[str, ...]
    all_pairs: tuple[tuple[str, str], ...]
    structural_interface_pairs: tuple[tuple[str, str], ...]
    forbidden_pairs: tuple[tuple[str, str], ...]
    all_pair_count: int
    structural_interface_pair_count: int
    forbidden_pair_count: int
    every_structural_interface_is_direct_urdf_parent_child: bool
    all_nonstructural_pairs_remain_forbidden: bool
    srdf_exemptions_applied: bool
    manual_pair_allowlist_used: bool
    online_truth_used: bool
    certificate_sha256: str

    def __post_init__(self) -> None:
        structural = tuple(self.structural_interface_pairs)
        forbidden = tuple(self.forbidden_pairs)
        structural_set = set(structural)
        expected_forbidden = tuple(
            pair for pair in self.all_pairs if pair not in structural_set
        )
        if (
            self.method_id != METHOD_ID
            or self.structural_interface_rule != STRUCTURAL_INTERFACE_RULE
            or not _is_sha256(self.aggregate_robot_model_sha256)
            or not _is_sha256(self.base_self_pair_inventory_sha256)
            or not _is_sha256(self.certificate_sha256)
            or self.link_names != tuple(sorted(self.link_names))
            or len(self.link_names) != EXPECTED_LINK_COUNT
            or self.all_pair_count != len(self.all_pairs)
            or self.all_pair_count != EXPECTED_BASE_PAIR_COUNT
            or self.structural_interface_pair_count != len(structural)
            or self.structural_interface_pair_count
            != EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT
            or self.forbidden_pair_count != len(forbidden)
            or self.forbidden_pair_count != EXPECTED_FORBIDDEN_PAIR_COUNT
            or self.all_pair_count
            != self.structural_interface_pair_count + self.forbidden_pair_count
            or structural != tuple(sorted(structural))
            or len(structural_set) != len(structural)
            or not structural_set <= set(self.all_pairs)
            or forbidden != expected_forbidden
            or self.every_structural_interface_is_direct_urdf_parent_child
            is not True
            or self.all_nonstructural_pairs_remain_forbidden is not True
            or any(
                value is not False
                for value in (
                    self.srdf_exemptions_applied,
                    self.manual_pair_allowlist_used,
                    self.online_truth_used,
                )
            )
            or self.certificate_sha256 != _certificate_sha256(self)
        ):
            raise ValueError("self-collision execution policy is malformed")

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": self.method_id,
                "structural_interface_rule": self.structural_interface_rule,
                "all_pair_count": self.all_pair_count,
                "structural_interface_pair_count": (
                    self.structural_interface_pair_count
                ),
                "forbidden_pair_count": self.forbidden_pair_count,
                "srdf_exemptions_applied": False,
                "manual_pair_allowlist_used": False,
                "online_truth_used": False,
                "certificate_sha256": self.certificate_sha256,
            }
        )


def build_self_collision_execution_policy(
    *,
    kinematic_binding: AggregateRobotKinematicBinding,
    base_inventory: SelfCollisionPairInventory,
) -> SelfCollisionExecutionPolicyCertificate:
    """Derive the only structural interfaces accepted by the route checker."""

    if type(kinematic_binding) is not AggregateRobotKinematicBinding:
        raise SelfCollisionExecutionPolicyError(
            "AGGREGATE_KINEMATIC_BINDING_REQUIRED",
            type(kinematic_binding).__name__,
        )
    if type(base_inventory) is not SelfCollisionPairInventory:
        raise SelfCollisionExecutionPolicyError(
            "BASE_SELF_COLLISION_INVENTORY_REQUIRED",
            type(base_inventory).__name__,
        )
    link_names = tuple(sorted(kinematic_binding.collision_link_names))
    if (
        link_names != base_inventory.link_names
        or kinematic_binding.model_sha256
        != kinematic_binding.model_sha256.lower()
    ):
        raise SelfCollisionExecutionPolicyError(
            "KINEMATIC_AND_PAIR_INVENTORY_MISMATCH",
            "collision link identities differ",
        )
    link_set = set(link_names)
    structural = tuple(
        sorted(
            {
                _canonical_pair(joint.parent_link, joint.child_link)
                for joint in kinematic_binding.model.joints.values()
                if joint.parent_link in link_set
                and joint.child_link in link_set
            }
        )
    )
    structural_set = set(structural)
    forbidden = tuple(
        pair for pair in base_inventory.all_pairs if pair not in structural_set
    )
    if (
        len(structural) != EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT
        or len(forbidden) != EXPECTED_FORBIDDEN_PAIR_COUNT
    ):
        raise SelfCollisionExecutionPolicyError(
            "STRUCTURAL_INTERFACE_COUNT_MISMATCH",
            f"structural={len(structural)}, forbidden={len(forbidden)}",
        )
    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "structural_interface_rule": STRUCTURAL_INTERFACE_RULE,
        "aggregate_robot_model_sha256": kinematic_binding.model_sha256,
        "base_self_pair_inventory_sha256": base_inventory.inventory_sha256,
        "link_names": link_names,
        "all_pairs": base_inventory.all_pairs,
        "structural_interface_pairs": structural,
        "forbidden_pairs": forbidden,
        "all_pair_count": len(base_inventory.all_pairs),
        "structural_interface_pair_count": len(structural),
        "forbidden_pair_count": len(forbidden),
        "every_structural_interface_is_direct_urdf_parent_child": True,
        "all_nonstructural_pairs_remain_forbidden": True,
        "srdf_exemptions_applied": False,
        "manual_pair_allowlist_used": False,
        "online_truth_used": False,
    }
    return SelfCollisionExecutionPolicyCertificate(
        **values,
        certificate_sha256=_certificate_sha256(values),
    )


__all__ = [
    "EXPECTED_BASE_PAIR_COUNT",
    "EXPECTED_FORBIDDEN_PAIR_COUNT",
    "EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT",
    "METHOD_ID",
    "STRUCTURAL_INTERFACE_RULE",
    "SelfCollisionExecutionPolicyCertificate",
    "SelfCollisionExecutionPolicyError",
    "build_self_collision_execution_policy",
]
