"""Fail-closed aggregation of complete sequential hand-surface checks.

The continuous-collision core proves one moving link surface against one
static object surface, or one moving link surface against one other moving
link surface.  This module binds those local certificates to an explicit
three-segment closure path and an exhaustive self-collision inventory.

V9 ray closure supplies finite interior PAD witnesses.  It does not supply a
continuous PAD-surface path certificate or an exact allowed endpoint patch.
Consequently this V1 aggregator can report that all presently checkable
non-PAD and self-collision gates passed, but it must remain
``NOT_CERTIFIABLE`` until a separate continuous PAD endpoint certificate is
defined.  No SRDF disabled-collision assertion is ever applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import json
import math
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.collision_contract import (
    SelfCollisionPairInventory,
    TerminalTrianglePartition,
    canonical_unoriented_triangle_bytes,
    triangle_instance_keys,
)
from kcg_connector.grasp.robust.continuous_collision import (
    ContinuousCollisionCertificate,
    ContinuousCollisionState,
    MovingSurfacePairCollisionCertificate,
    certify_moving_link_surfaces_separated_from_each_other,
    certify_moving_link_surface_separated_from_static_surface,
)
from kcg_connector.grasp.robust.hand_contract import (
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DirectedIntervalKinematics,
    IntervalBounds,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
)
from kcg_connector.grasp.robust.object_model import load_stl_mesh
from kcg_connector.grasp.robust.ray_closure import (
    CLOSURE_FOCUS_METHOD,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CLOSURE_SUFFIX_DOMINANCE_ARGUMENT,
    FEATURE_ROOT_POLICY,
    INTERVAL_RULE,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX,
    RAY_EVALUATION_POLICY,
    RayClosureEvaluation,
    WITNESS_RULE,
)


METHOD_ID = "CARTS_FULL_HAND_SEQUENTIAL_CLOSURE_COLLISION_AGGREGATOR_V1"
SURFACE_HASH_METHOD_ID = (
    "CARTS_UNORIENTED_UNORDERED_TRIANGLE_SURFACE_V1"
)
ALLOWED_V9_CONTACT_CLASSIFICATION = (
    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
)
PAD_SURFACE_BLOCKER_PREFIX = (
    "PAD_SURFACE_CONTINUOUS_PATH_AND_ENDPOINT_CERTIFICATE_UNAVAILABLE"
)
CLAIM_LIMITATIONS = (
    "SEQUENTIAL_THREE_FINGER_CLOSURE_PATH_ONLY",
    "MOVING_SURFACE_VS_STATIC_OBJECT_SURFACE_AND_SELF_SURFACE_PAIRS_ONLY",
    "NOT_SOLID_CONTAINMENT_OR_INTERIOR_EXCLUSION",
    "NOT_ENVIRONMENT_OR_ARM_PATH_COLLISION",
    "NO_SRDF_NEVER_OR_ADJACENT_EXEMPTIONS_APPLIED",
    "V9_FINITE_PAD_WITNESSES_ARE_NOT_CONTINUOUS_PAD_SURFACE_CCD",
    "NO_EXACT_ALLOWED_PAD_ENDPOINT_PATCH_CERTIFICATE_IS_AVAILABLE",
    "AUTHORITATIVE_FULL_HAND_COLLISION_LINK_ROSTER_NOT_BOUND",
    "POTENTIAL_CONTACT_TANGENCY_AND_COPLANARITY_REMAIN_UNRESOLVED",
)


class FullHandCollisionError(ValueError):
    """Raised when an explicit aggregation contract is malformed."""


class FullHandClosureCollisionState(str, Enum):
    """Formal state of the aggregate collision claim."""

    CERTIFIED = "CERTIFIED"
    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _normalise_signed_zero(array: np.ndarray) -> np.ndarray:
    return np.where(array == 0.0, 0.0, array)


def _float64_hex(value: float) -> str:
    return float(value).hex()


def _float64_array_hex(value: object) -> list[object]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return [_float64_hex(float(array))]
    if array.ndim == 1:
        return [_float64_hex(float(item)) for item in array]
    return [_float64_array_hex(row) for row in array]


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _binary64_array_equal(first: object, second: object) -> bool:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    return bool(
        first_array.shape == second_array.shape
        and np.asarray(first_array, dtype="<f8").tobytes(order="C")
        == np.asarray(second_array, dtype="<f8").tobytes(order="C")
    )


def _decode_float64_hex_matrix(
    value: object,
    *,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != shape[0]
        or any(
            not isinstance(row, list) or len(row) != shape[1]
            for row in value
        )
    ):
        raise FullHandCollisionError(
            f"{label} must have canonical matrix shape {shape}"
        )
    try:
        result = np.asarray(
            [
                [float.fromhex(token) for token in row]
                for row in value
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise FullHandCollisionError(
            f"{label} contains a non-binary64-hex token"
        ) from error
    if (
        not np.all(np.isfinite(result))
        or _float64_array_hex(result) != value
    ):
        raise FullHandCollisionError(
            f"{label} is not a canonical finite binary64 matrix"
        )
    return result


def _joint_model_manifest(hand_model: object) -> list[object]:
    rows: list[object] = []
    for mapping_key in sorted(hand_model.joints):
        joint = hand_model.joints[mapping_key]
        limit = (
            None
            if joint.limit is None
            else {
                "lower": _float64_hex(joint.limit.lower),
                "upper": _float64_hex(joint.limit.upper),
                "effort": (
                    None
                    if joint.limit.effort is None
                    else _float64_hex(joint.limit.effort)
                ),
                "velocity": (
                    None
                    if joint.limit.velocity is None
                    else _float64_hex(joint.limit.velocity)
                ),
            }
        )
        mimic = (
            None
            if joint.mimic is None
            else {
                "source_joint": joint.mimic.source_joint,
                "multiplier": _float64_hex(joint.mimic.multiplier),
                "offset": _float64_hex(joint.mimic.offset),
            }
        )
        rows.append(
            {
                "mapping_key": mapping_key,
                "name": joint.name,
                "joint_type": joint.joint_type,
                "parent_link": joint.parent_link,
                "child_link": joint.child_link,
                "origin_xyz_m": _float64_array_hex(joint.origin_xyz_m),
                "origin_rpy_rad": _float64_array_hex(
                    joint.origin_rpy_rad
                ),
                "axis": _float64_array_hex(joint.axis),
                "limit": limit,
                "mimic": mimic,
            }
        )
    return rows


def _hand_pad_model_manifest(hand_model: object) -> list[object]:
    rows: list[object] = []
    for name in sorted(hand_model.pads):
        pad = hand_model.pads[name]
        rows.append(
            {
                "mapping_key": name,
                "name": pad.name,
                "finger_name": pad.finger_name,
                "link_name": pad.link_name,
                "origin_xyz_m": _float64_array_hex(pad.origin_xyz_m),
                "origin_rpy_rad": _float64_array_hex(pad.origin_rpy_rad),
                "geometry": {
                    "kind": pad.geometry.kind,
                    "dimensions_m": _float64_array_hex(
                        pad.geometry.dimensions_m
                    ),
                    "mesh_uri": pad.geometry.mesh_uri,
                    "mesh_scale": _float64_array_hex(
                        pad.geometry.mesh_scale
                    ),
                },
                "contact_normal_pad": (
                    None
                    if pad.contact_normal_pad is None
                    else _float64_array_hex(pad.contact_normal_pad)
                ),
                "normal_force_capacity_n": (
                    None
                    if pad.normal_force_capacity_n is None
                    else _float64_hex(pad.normal_force_capacity_n)
                ),
            }
        )
    return rows


def _hand_model_manifest(hand_model: object) -> dict[str, object]:
    independent_limits = hand_model.independent_joint_limits
    return {
        "base_link": hand_model.base_link,
        "joint_order": list(hand_model.joint_order),
        "independent_joint_names": list(
            hand_model.independent_joint_names
        ),
        "joints_by_mapping_key": _joint_model_manifest(hand_model),
        "independent_affine_limits": [
            {
                "joint_name": name,
                "lower": _float64_hex(independent_limits[name].lower),
                "upper": _float64_hex(independent_limits[name].upper),
                "effort": (
                    None
                    if independent_limits[name].effort is None
                    else _float64_hex(independent_limits[name].effort)
                ),
                "velocity": (
                    None
                    if independent_limits[name].velocity is None
                    else _float64_hex(independent_limits[name].velocity)
                ),
            }
            for name in hand_model.independent_joint_names
        ],
        "finger_chains": [
            {
                "finger_name": name,
                "joint_names": list(hand_model.fingers[name].joint_names),
                "terminal_link": hand_model.fingers[name].terminal_link,
                "pad_name": hand_model.fingers[name].pad_name,
            }
            for name in sorted(hand_model.fingers)
        ],
        "pad_mapping": _hand_pad_model_manifest(hand_model),
    }


def _surface_geometry(
    triangles_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, str]:
    """Validate and hash the same surface identity used by the CCD core."""

    triangles = np.asarray(triangles_m, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or not np.all(np.isfinite(triangles))
    ):
        raise FullHandCollisionError(
            f"{label} must be a non-empty finite array with shape (F, 3, 3)"
        )
    rows: list[bytes] = []
    for triangle in triangles:
        canonical = _normalise_signed_zero(triangle)
        order = sorted(
            range(3),
            key=lambda index: tuple(
                float(value) for value in canonical[index]
            ),
        )
        canonical = canonical[np.asarray(order, dtype=np.int64)]
        with np.errstate(over="ignore", invalid="ignore"):
            edges = canonical[1:] - canonical[0]
        if not np.all(np.isfinite(edges)):
            raise FullHandCollisionError(
                f"{label} triangle edge arithmetic overflowed"
            )
        edge_scale = float(np.max(np.abs(edges)))
        if edge_scale == 0.0:
            raise FullHandCollisionError(
                f"{label} contains an exactly degenerate triangle"
            )
        scaled_cross = np.cross(
            edges[0] / edge_scale,
            edges[1] / edge_scale,
        )
        if (
            not np.all(np.isfinite(scaled_cross))
            or float(np.linalg.norm(scaled_cross)) == 0.0
        ):
            raise FullHandCollisionError(
                f"{label} contains an exactly degenerate triangle"
            )
        rows.append(
            np.asarray(canonical, dtype="<f8").tobytes(order="C")
        )
    rows.sort()
    digest = hashlib.sha256()
    digest.update(SURFACE_HASH_METHOD_ID.encode("ascii") + b"\0")
    digest.update(np.asarray((len(rows),), dtype="<u8").tobytes())
    for row in rows:
        digest.update(row)
    # A NumPy-owned read-only array can be made writable again by its owner.
    # A bytes-backed view is a deep immutable snapshot of the supplied faces.
    snapshot = np.asarray(triangles, dtype="<f8").tobytes(order="C")
    result = np.frombuffer(snapshot, dtype="<f8").reshape(triangles.shape)
    return result, digest.hexdigest()


def triangle_surface_geometry_sha256(
    triangles_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
) -> str:
    """Return the face-order, winding, and signed-zero invariant core hash."""

    _triangles, digest = _surface_geometry(
        triangles_m,
        label="triangle surface",
    )
    return digest


def _exact_unoriented_triangle_multiset(
    triangles_m: np.ndarray,
) -> tuple[bytes, ...]:
    return tuple(
        sorted(
            canonical_unoriented_triangle_bytes(triangle)
            for triangle in triangles_m
        )
    )


@dataclass(frozen=True)
class HashBoundLinkSurface:
    """One complete link-local triangle surface and its exact digests."""

    link_name: str
    source_asset_sha256: str
    geometry_sha256: str
    triangles_link_m: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.link_name):
            raise FullHandCollisionError("link surface needs a link_name")
        if not _valid_sha256(self.source_asset_sha256):
            raise FullHandCollisionError(
                "link surface source_asset_sha256 is invalid"
            )
        triangles, digest = _surface_geometry(
            self.triangles_link_m,
            label=f"link surface {self.link_name}",
        )
        if self.geometry_sha256 != digest:
            raise FullHandCollisionError(
                f"link surface geometry hash mismatch: {self.link_name}"
            )
        object.__setattr__(self, "link_name", str(self.link_name))
        object.__setattr__(self, "triangles_link_m", triangles)


@dataclass(frozen=True)
class HashBoundObjectSurface:
    """Static object-frame triangle surface bound to source and geometry."""

    object_id: str
    source_asset_sha256: str
    geometry_sha256: str
    ray_closure_object_geometry_sha256: str
    triangles_object_m: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.object_id):
            raise FullHandCollisionError("object surface needs an object_id")
        if not _valid_sha256(self.source_asset_sha256) or not _valid_sha256(
            self.ray_closure_object_geometry_sha256
        ):
            raise FullHandCollisionError(
                "object surface source/Ray geometry SHA-256 is invalid"
            )
        triangles, digest = _surface_geometry(
            self.triangles_object_m,
            label=f"object surface {self.object_id}",
        )
        if self.geometry_sha256 != digest:
            raise FullHandCollisionError(
                f"object surface geometry hash mismatch: {self.object_id}"
            )
        object.__setattr__(self, "object_id", str(self.object_id))
        object.__setattr__(self, "triangles_object_m", triangles)


@dataclass(frozen=True)
class TerminalForbiddenSurface:
    """Exact non-PAD subset supplied for one terminal link."""

    link_name: str
    partition: TerminalTrianglePartition
    nonpad_forbidden_surface: HashBoundLinkSurface

    def __post_init__(self) -> None:
        if not str(self.link_name):
            raise FullHandCollisionError(
                "terminal forbidden surface needs a link_name"
            )
        if not isinstance(self.partition, TerminalTrianglePartition):
            raise FullHandCollisionError(
                "terminal forbidden surface needs TerminalTrianglePartition"
            )
        if not isinstance(
            self.nonpad_forbidden_surface, HashBoundLinkSurface
        ):
            raise FullHandCollisionError(
                "terminal forbidden surface is not hash-bound"
            )
        if self.nonpad_forbidden_surface.link_name != self.link_name:
            raise FullHandCollisionError(
                "terminal forbidden surface link binding disagrees"
            )
        object.__setattr__(self, "link_name", str(self.link_name))


@dataclass(frozen=True)
class SequentialClosureSegment:
    """One explicitly registered scalar segment of sequential closure."""

    segment_index: int
    pad_name: str
    active_link_name: str
    q_start: tuple[float, ...]
    direction: tuple[float, ...]
    phase: IntervalBounds
    maximum_subdivision_intervals: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_index, int)
            or isinstance(self.segment_index, bool)
            or self.segment_index < 0
        ):
            raise FullHandCollisionError(
                "closure segment index must be a nonnegative integer"
            )
        if not str(self.pad_name) or not str(self.active_link_name):
            raise FullHandCollisionError(
                "closure segment must bind one PAD and active link"
            )
        start = tuple(float(value) for value in self.q_start)
        direction = tuple(float(value) for value in self.direction)
        if (
            not start
            or len(start) != len(direction)
            or not all(math.isfinite(value) for value in start + direction)
        ):
            raise FullHandCollisionError(
                "closure segment q_start/direction are malformed"
            )
        if not isinstance(self.phase, IntervalBounds):
            raise FullHandCollisionError(
                "closure segment phase must be explicit IntervalBounds"
            )
        if (
            not isinstance(self.maximum_subdivision_intervals, int)
            or isinstance(self.maximum_subdivision_intervals, bool)
            or self.maximum_subdivision_intervals <= 0
        ):
            raise FullHandCollisionError(
                "closure segment subdivision budget must be positive"
            )
        object.__setattr__(self, "pad_name", str(self.pad_name))
        object.__setattr__(
            self, "active_link_name", str(self.active_link_name)
        )
        object.__setattr__(self, "q_start", start)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class LinkObjectPathCertificate:
    segment_index: int
    link_name: str
    collision_domain: str
    certificate: ContinuousCollisionCertificate


@dataclass(frozen=True)
class SelfPairPathCertificate:
    segment_index: int
    first_link_name: str
    second_link_name: str
    certificate: MovingSurfacePairCollisionCertificate


@dataclass(frozen=True)
class FullHandClosureCollisionAudit:
    method_id: str
    interval_kinematics_method_id: str
    ray_closure_method_id: str
    v9_evidence_sha256: str
    object_id: str
    object_source_asset_sha256: str
    object_surface_geometry_sha256: str
    ray_closure_object_geometry_sha256: str
    ray_model_contract_sha256: str
    link_surface_bindings: tuple[tuple[str, str, str], ...]
    terminal_partition_bindings: tuple[tuple[str, str, str], ...]
    self_pair_inventory_sha256: str
    segment_contract_sha256: tuple[str, ...]
    segment_budget_usage: tuple[tuple[int, int, int], ...]
    segment_count: int
    link_count: int
    terminal_link_count: int
    self_pair_count_per_segment: int
    expected_link_object_domain_count: int
    evaluated_link_object_domain_count: int
    certified_free_link_object_domain_count: int
    expected_self_pair_domain_count: int
    evaluated_self_pair_domain_count: int
    certified_free_self_pair_domain_count: int
    all_link_object_domains_covered: bool
    all_self_pair_domains_covered: bool
    srdf_exemptions_applied: bool
    pad_endpoint_continuous_surface_certificate_present: bool
    checkable_collision_gates_passed: bool
    blockers: tuple[str, ...]
    claim_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            any(len(row) != 3 for row in self.link_surface_bindings)
            or any(
                len(row) != 3 for row in self.terminal_partition_bindings
            )
        ):
            raise FullHandCollisionError(
                "full-hand surface binding rows are malformed"
            )
        digests = (
            self.v9_evidence_sha256,
            self.object_source_asset_sha256,
            self.object_surface_geometry_sha256,
            self.ray_closure_object_geometry_sha256,
            self.ray_model_contract_sha256,
            self.self_pair_inventory_sha256,
            *self.segment_contract_sha256,
            *(row[1] for row in self.link_surface_bindings),
            *(row[2] for row in self.link_surface_bindings),
            *(row[1] for row in self.terminal_partition_bindings),
            *(row[2] for row in self.terminal_partition_bindings),
        )
        if self.method_id != METHOD_ID:
            raise FullHandCollisionError("full-hand method identifier changed")
        if self.interval_kinematics_method_id != (
            INTERVAL_KINEMATICS_METHOD_ID
        ):
            raise FullHandCollisionError(
                "full-hand interval backend identifier changed"
            )
        if self.ray_closure_method_id != RAY_CLOSURE_METHOD_ID:
            raise FullHandCollisionError(
                "full-hand V9 evidence identifier changed"
            )
        if any(not _valid_sha256(value) for value in digests):
            raise FullHandCollisionError(
                "full-hand audit contains an invalid digest"
            )
        link_names = tuple(row[0] for row in self.link_surface_bindings)
        terminal_names = tuple(
            row[0] for row in self.terminal_partition_bindings
        )
        if (
            not self.object_id
            or link_names != tuple(sorted(link_names))
            or len(set(link_names)) != len(link_names)
            or terminal_names != tuple(sorted(terminal_names))
            or len(set(terminal_names)) != len(terminal_names)
            or not set(terminal_names) <= set(link_names)
            or self.blockers != tuple(sorted(set(self.blockers)))
        ):
            raise FullHandCollisionError(
                "full-hand identifiers or blocker order are not canonical"
            )
        integer_fields = (
            self.segment_count,
            self.link_count,
            self.terminal_link_count,
            self.self_pair_count_per_segment,
            self.expected_link_object_domain_count,
            self.evaluated_link_object_domain_count,
            self.certified_free_link_object_domain_count,
            self.expected_self_pair_domain_count,
            self.evaluated_self_pair_domain_count,
            self.certified_free_self_pair_domain_count,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in integer_fields
        ):
            raise FullHandCollisionError(
                "full-hand audit counters must be nonnegative integers"
            )
        if (
            self.segment_count != 3
            or self.segment_count != len(self.segment_contract_sha256)
            or self.link_count != len(self.link_surface_bindings)
            or self.terminal_link_count
            != len(self.terminal_partition_bindings)
            or self.terminal_link_count != 3
            or self.self_pair_count_per_segment
            != self.link_count * (self.link_count - 1) // 2
            or self.expected_link_object_domain_count
            != self.segment_count * self.link_count
            or self.expected_self_pair_domain_count
            != self.segment_count * self.self_pair_count_per_segment
            or self.certified_free_link_object_domain_count
            > self.evaluated_link_object_domain_count
            or self.certified_free_self_pair_domain_count
            > self.evaluated_self_pair_domain_count
        ):
            raise FullHandCollisionError(
                "full-hand audit coverage arithmetic is inconsistent"
            )
        if (
            len(self.segment_budget_usage) != self.segment_count
            or any(
                maximum <= 0
                or used < 0
                or remaining < 0
                or used + remaining != maximum
                for maximum, used, remaining in self.segment_budget_usage
            )
        ):
            raise FullHandCollisionError(
                "full-hand shared segment budget accounting is inconsistent"
            )
        if self.all_link_object_domains_covered != (
            self.evaluated_link_object_domain_count
            == self.expected_link_object_domain_count
        ) or self.all_self_pair_domains_covered != (
            self.evaluated_self_pair_domain_count
            == self.expected_self_pair_domain_count
        ):
            raise FullHandCollisionError(
                "full-hand audit coverage flags are inconsistent"
            )
        expected_checkable = (
            self.all_link_object_domains_covered
            and self.all_self_pair_domains_covered
            and self.certified_free_link_object_domain_count
            == self.expected_link_object_domain_count
            and self.certified_free_self_pair_domain_count
            == self.expected_self_pair_domain_count
        )
        if self.checkable_collision_gates_passed != expected_checkable:
            raise FullHandCollisionError(
                "full-hand checkable-gate flag is inconsistent"
            )
        if (
            self.srdf_exemptions_applied
            or self.pad_endpoint_continuous_surface_certificate_present
        ):
            raise FullHandCollisionError(
                "V1 cannot apply SRDF exemptions or claim PAD endpoint "
                "coverage"
            )
        if not self.blockers or self.claim_limitations != CLAIM_LIMITATIONS:
            raise FullHandCollisionError(
                "full-hand blockers or limitations were altered"
            )


@dataclass(frozen=True)
class FullHandClosureCollisionCertificate:
    state: FullHandClosureCollisionState
    link_object_certificates: tuple[LinkObjectPathCertificate, ...]
    self_pair_certificates: tuple[SelfPairPathCertificate, ...]
    audit: FullHandClosureCollisionAudit

    def __post_init__(self) -> None:
        if not isinstance(self.state, FullHandClosureCollisionState):
            raise FullHandCollisionError("full-hand state is invalid")
        object.__setattr__(
            self,
            "link_object_certificates",
            tuple(self.link_object_certificates),
        )
        object.__setattr__(
            self,
            "self_pair_certificates",
            tuple(self.self_pair_certificates),
        )
        if len(self.link_object_certificates) != (
            self.audit.evaluated_link_object_domain_count
        ) or len(self.self_pair_certificates) != (
            self.audit.evaluated_self_pair_domain_count
        ):
            raise FullHandCollisionError(
                "full-hand child certificate counts are inconsistent"
            )
        link_hashes = {
            name: geometry
            for name, _source, geometry in self.audit.link_surface_bindings
        }
        terminal_hashes = {
            name: geometry
            for name, _partition, geometry in (
                self.audit.terminal_partition_bindings
            )
        }
        link_names = tuple(link_hashes)
        expected_object_keys = {
            (segment_index, link_name)
            for segment_index in range(self.audit.segment_count)
            for link_name in link_names
        }
        object_keys = tuple(
            (row.segment_index, row.link_name)
            for row in self.link_object_certificates
        )
        if (
            object_keys != tuple(sorted(object_keys))
            or len(set(object_keys)) != len(object_keys)
            or not set(object_keys) <= expected_object_keys
        ):
            raise FullHandCollisionError(
                "link/object child records are duplicated or out of domain"
            )
        for row in self.link_object_certificates:
            expected_terminal = row.link_name in terminal_hashes
            expected_hash = (
                terminal_hashes[row.link_name]
                if expected_terminal
                else link_hashes[row.link_name]
            )
            expected_domain = (
                "TERMINAL_EXACT_NONPAD_FORBIDDEN_SURFACE"
                if expected_terminal
                else "FULL_LINK_SURFACE"
            )
            child_audit = row.certificate.audit
            if (
                not isinstance(row.certificate, ContinuousCollisionCertificate)
                or row.collision_domain != expected_domain
                or child_audit.link_name != row.link_name
                or child_audit.moving_surface_geometry_sha256
                != expected_hash
                or child_audit.static_surface_geometry_sha256
                != self.audit.object_surface_geometry_sha256
            ):
                raise FullHandCollisionError(
                    "link/object child certificate binding is inconsistent"
                )
        expected_pairs = tuple(itertools.combinations(link_names, 2))
        expected_self_keys = {
            (segment_index, first, second)
            for segment_index in range(self.audit.segment_count)
            for first, second in expected_pairs
        }
        self_keys = tuple(
            (
                row.segment_index,
                row.first_link_name,
                row.second_link_name,
            )
            for row in self.self_pair_certificates
        )
        if (
            self_keys != tuple(sorted(self_keys))
            or len(set(self_keys)) != len(self_keys)
            or not set(self_keys) <= expected_self_keys
        ):
            raise FullHandCollisionError(
                "self-pair child records are duplicated or out of domain"
            )
        for row in self.self_pair_certificates:
            child_audit = row.certificate.audit
            if (
                not isinstance(
                    row.certificate,
                    MovingSurfacePairCollisionCertificate,
                )
                or child_audit.first_link_name != row.first_link_name
                or child_audit.second_link_name != row.second_link_name
                or child_audit.first_surface_geometry_sha256
                != link_hashes[row.first_link_name]
                or child_audit.second_surface_geometry_sha256
                != link_hashes[row.second_link_name]
            ):
                raise FullHandCollisionError(
                    "self-pair child certificate binding is inconsistent"
                )
        recomputed_free_object = sum(
            row.certificate.state is ContinuousCollisionState.CERTIFIED_FREE
            for row in self.link_object_certificates
        )
        recomputed_free_self = sum(
            row.certificate.state is ContinuousCollisionState.CERTIFIED_FREE
            for row in self.self_pair_certificates
        )
        if (
            recomputed_free_object
            != self.audit.certified_free_link_object_domain_count
            or recomputed_free_self
            != self.audit.certified_free_self_pair_domain_count
        ):
            raise FullHandCollisionError(
                "full-hand free child counts were not recomputed correctly"
            )
        if self.state is FullHandClosureCollisionState.CERTIFIED:
            if (
                self.audit.blockers
                or not self.audit.checkable_collision_gates_passed
                or not self.audit.
                pad_endpoint_continuous_surface_certificate_present
            ):
                raise FullHandCollisionError(
                    "certified full-hand state lacks complete evidence"
                )
        elif not self.audit.blockers:
            raise FullHandCollisionError(
                "not-certifiable state must preserve its blockers"
            )


def _segment_sha256(segment: SequentialClosureSegment) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0SEGMENT\0")
    digest.update(segment.segment_index.to_bytes(8, "little"))
    for value in (segment.pad_name, segment.active_link_name):
        digest.update(value.encode("utf-8") + b"\0")
    for vector in (
        segment.q_start,
        segment.direction,
        (segment.phase.lower, segment.phase.upper),
    ):
        digest.update(
            np.asarray(
                _normalise_signed_zero(
                    np.asarray(vector, dtype=np.float64)
                ),
                dtype="<f8",
            ).tobytes()
        )
    digest.update(
        segment.maximum_subdivision_intervals.to_bytes(8, "little")
    )
    return digest.hexdigest()


def _v9_evidence_sha256(evaluation: RayClosureEvaluation) -> str:
    candidate = evaluation.candidate
    if candidate is None:
        raise FullHandCollisionError("V9 evaluation has no candidate")
    payload = {
        "audit": evaluation.audit.as_dict(),
        "candidate": {
            "object_from_hand": list(candidate.object_from_hand),
            "independent_joint_positions_rad": list(
                candidate.independent_joint_positions_rad
            ),
            "planned_pad_contacts": [
                {
                    "pad_name": row.pad_name,
                    "position_object_m": list(row.position_object_m),
                    "path_local_free_side_normal_object": list(
                        row.path_local_free_side_normal_object
                    ),
                    "surface_coordinates": list(row.surface_coordinates),
                }
                for row in candidate.planned_pad_contacts
            ],
            "internal_normal_forces_n": list(
                candidate.internal_normal_forces_n
            ),
            "stiffness_diagonal": list(candidate.stiffness_diagonal),
            "damping_diagonal": list(candidate.damping_diagonal),
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terminal_pad_runtime_geometry_evidence(
    partition: TerminalTrianglePartition,
) -> tuple[str, int, int]:
    try:
        with np.load(partition.pad_source_path, allow_pickle=False) as rows:
            points = np.asarray(rows["points_local_m"], dtype=np.float64)
            faces_input = np.asarray(rows["faces"])
    except (KeyError, OSError, ValueError) as error:
        raise FullHandCollisionError(
            "terminal PAD source cannot be loaded for runtime binding"
        ) from error
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not np.all(np.isfinite(points))
        or faces_input.ndim != 2
        or faces_input.shape[1:] != (3,)
        or faces_input.dtype.kind not in "iu"
    ):
        raise FullHandCollisionError(
            "terminal PAD runtime arrays are malformed"
        )
    faces = np.asarray(faces_input, dtype=np.int64)
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise FullHandCollisionError(
            "terminal PAD runtime face index is outside points"
        )
    digest = hashlib.sha256()
    digest.update(b"CARTS_VERIFIED_PAD_RUNTIME_TRIANGLE_MESH_SI_V1\0")
    for value, dtype in (
        (points, np.dtype("<f8")),
        (faces, np.dtype("<i8")),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        digest.update(
            np.asarray((array.ndim,), dtype="<i8").tobytes()
        )
        digest.update(
            np.asarray(array.shape, dtype="<i8").tobytes(order="C")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest(), len(points), len(faces)


def _validate_v9_model_binding(
    *,
    backend: DirectedIntervalKinematics,
    evaluation: RayClosureEvaluation,
    segments: tuple[SequentialClosureSegment, ...],
    object_surface: HashBoundObjectSurface,
    terminal_surfaces: tuple[TerminalForbiddenSurface, ...],
) -> None:
    audit = evaluation.audit
    if (
        audit.model_binding_complete is not True
        or audit.model_binding_status != MODEL_BINDING_COMPLETE_STATUS
    ):
        raise FullHandCollisionError(
            "V9 model binding is synthetic, unbound, or incomplete"
        )
    if audit.object_geometry_sha256 != (
        object_surface.ray_closure_object_geometry_sha256
    ):
        raise FullHandCollisionError(
            "V9 object geometry differs from the hash-bound object surface"
        )
    try:
        document = json.loads(audit.model_contract_canonical_json)
    except (TypeError, ValueError) as error:
        raise FullHandCollisionError(
            "V9 model contract canonical JSON cannot be decoded"
        ) from error
    if not isinstance(document, Mapping):
        raise FullHandCollisionError(
            "V9 model contract document must be a mapping"
        )
    canonical_json = _canonical_json(document)
    recomputed_contract_sha256 = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    if (
        canonical_json != audit.model_contract_canonical_json
        or recomputed_contract_sha256 != audit.model_contract_sha256
    ):
        raise FullHandCollisionError(
            "V9 model contract canonical JSON digest does not recompute"
        )
    hand_model = backend.hand_model
    joint_names = tuple(hand_model.independent_joint_names)
    expected_physical = np.asarray(
        tuple(segment.direction for segment in segments),
        dtype=np.float64,
    )
    try:
        closure_document = document["closure"]
        ray_document = document["ray_closure"]
        interval_document = document["interval_backend"]
        pad_rows = document["verified_pads"]
        object_document = document["object"]
        if (
            document["schema"] != MODEL_CONTRACT_DIGEST_METHOD_ID
            or document["hand"] != _hand_model_manifest(hand_model)
            or object_document["geometry_sha256"]
            != object_surface.ray_closure_object_geometry_sha256
            or not isinstance(pad_rows, list)
        ):
            raise FullHandCollisionError(
                "V9 model contract object or complete hand binding differs"
            )
    except (KeyError, TypeError) as error:
        raise FullHandCollisionError(
            "V9 model contract is structurally incomplete"
        ) from error
    production_ray_fields = {
        "method_id": RAY_CLOSURE_METHOD_ID,
        "closure_parameter_domain_id": CLOSURE_PARAMETER_DOMAIN_ID,
        "closure_focus_method": CLOSURE_FOCUS_METHOD,
        "feature_root_policy": FEATURE_ROOT_POLICY,
        "ray_evaluation_policy": RAY_EVALUATION_POLICY,
        "witness_rule": WITNESS_RULE,
        "interval_rule": INTERVAL_RULE,
        "object_contact_normal_policy": OBJECT_CONTACT_NORMAL_POLICY,
        "pad_surface_normal_policy": PAD_SURFACE_NORMAL_POLICY,
        "maximum_subdivision_intervals": (
            audit.maximum_subdivision_intervals
        ),
    }
    if (
        any(
            ray_document.get(name) != value
            for name, value in production_ray_fields.items()
        )
        or audit.witness_rule != WITNESS_RULE
        or audit.interval_rule != INTERVAL_RULE
        or audit.ray_evaluation_policy != RAY_EVALUATION_POLICY
        or audit.feature_root_policy != FEATURE_ROOT_POLICY
        or audit.object_contact_normal_policy
        != OBJECT_CONTACT_NORMAL_POLICY
        or audit.pad_surface_normal_policy != PAD_SURFACE_NORMAL_POLICY
        or audit.closure_parameter_domain_id
        != CLOSURE_PARAMETER_DOMAIN_ID
        or audit.closure_suffix_dominance_argument
        != CLOSURE_SUFFIX_DOMINANCE_ARGUMENT
        or audit.closure_focus_method != CLOSURE_FOCUS_METHOD
    ):
        raise FullHandCollisionError(
            "V9 audit is not the registered production method contract"
        )
    if (
        interval_document.get("method_id")
        != INTERVAL_KINEMATICS_METHOD_ID
        or interval_document.get("decimal_precision")
        != backend.options.decimal_precision
        or interval_document.get("maximum_root_bisection_iterations")
        != backend.options.maximum_root_bisection_iterations
        or audit.interval_arithmetic_method_id
        != INTERVAL_KINEMATICS_METHOD_ID
        or audit.interval_decimal_precision
        != backend.options.decimal_precision
        or audit.maximum_root_bisection_iterations
        != backend.options.maximum_root_bisection_iterations
    ):
        raise FullHandCollisionError(
            "V9 interval backend/options differ from the shared backend"
        )
    expected_supports = tuple(
        tuple(row) for row in audit.independent_actuation_supports
    )
    expected_parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}"
        for name in audit.preshape_joint_names
    )
    if (
        tuple(audit.parameter_layout) != expected_parameter_layout
        or closure_document.get("parameter_layout")
        != list(expected_parameter_layout)
        or closure_document.get("independent_actuation_supports")
        != [list(row) for row in expected_supports]
        or closure_document.get("closing_directions_physical")
        != _float64_array_hex(expected_physical)
        or not _binary64_array_equal(
            audit.closing_directions_physical,
            expected_physical,
        )
    ):
        raise FullHandCollisionError(
            "V9 closure supports/directions differ from the explicit path"
        )
    unit_directions = _decode_float64_hex_matrix(
        closure_document.get("closing_directions_unit"),
        shape=(3, len(joint_names)),
        label="V9 unit closing directions",
    )
    lower, upper = hand_model.joint_limit_vectors()
    spans = upper - lower
    recomputed_physical = unit_directions * spans
    used_support_indices: set[int] = set()
    recomputed_open: list[float] = []
    for index, (unit_row, support_names) in enumerate(
        zip(unit_directions, expected_supports)
    ):
        support_indices = tuple(
            row for row, value in enumerate(unit_row) if value != 0.0
        )
        if (
            len(support_indices) != 1
            or len(support_names) != 1
            or joint_names[support_indices[0]] != support_names[0]
            or support_indices[0] in used_support_indices
            or float(np.max(np.abs(unit_row))) != 1.0
        ):
            raise FullHandCollisionError(
                f"V9 closing direction {index} is not its exclusive "
                "canonical support"
            )
        support_index = support_indices[0]
        used_support_indices.add(support_index)
        recomputed_open.append(
            float(
                lower[support_index]
                if unit_row[support_index] > 0.0
                else upper[support_index]
            )
        )
    expected_preshape_names = tuple(
        name
        for index, name in enumerate(joint_names)
        if index not in used_support_indices and spans[index] > 0.0
    )
    if (
        not _binary64_array_equal(recomputed_physical, expected_physical)
        or not _binary64_array_equal(
            recomputed_open,
            audit.closure_open_joint_positions_rad,
        )
        or audit.preshape_joint_names != expected_preshape_names
    ):
        raise FullHandCollisionError(
            "V9 directions do not recompute from hand limits and open supports"
        )
    terminal_by_link = {row.link_name: row for row in terminal_surfaces}
    expected_pad_source_hashes: list[str] = []
    expected_pad_runtime_hashes: list[str] = []
    expected_pad_links: list[str] = []
    if len(pad_rows) != 3:
        raise FullHandCollisionError(
            "V9 model contract does not contain exactly three PAD rows"
        )
    for index, (segment, pad_row) in enumerate(zip(segments, pad_rows)):
        if not isinstance(pad_row, Mapping):
            raise FullHandCollisionError(
                "V9 PAD model row is not a mapping"
            )
        terminal = terminal_by_link[segment.active_link_name]
        runtime_hash, vertex_count, triangle_count = (
            _terminal_pad_runtime_geometry_evidence(terminal.partition)
        )
        hand_pad = hand_model.pads[segment.pad_name]
        expected_pad_source_hashes.append(
            terminal.partition.pad_source_sha256
        )
        expected_pad_runtime_hashes.append(runtime_hash)
        expected_pad_links.append(segment.active_link_name)
        expected_values = {
            "name": segment.pad_name,
            "finger_name": hand_pad.finger_name,
            "link_name": segment.active_link_name,
            "coordinate_frame": segment.active_link_name,
            "unit": "m",
            "source_mesh_sha256": terminal.partition.pad_source_sha256,
            "source_mesh_byte_count": (
                terminal.partition.pad_source_path.stat().st_size
            ),
            "runtime_geometry_sha256": runtime_hash,
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
        }
        if any(
            pad_row.get(name) != value
            for name, value in expected_values.items()
        ):
            raise FullHandCollisionError(
                f"V9 PAD source/runtime/link binding differs at row {index}"
            )
    if (
        audit.pad_geometry_sha256 != tuple(expected_pad_source_hashes)
        or audit.pad_runtime_geometry_sha256
        != tuple(expected_pad_runtime_hashes)
        or audit.pad_link_names != tuple(expected_pad_links)
    ):
        raise FullHandCollisionError(
            "V9 PAD audit hashes/links differ from terminal partitions"
        )


def _validate_v9_path_binding(
    *,
    backend: DirectedIntervalKinematics,
    evaluation: RayClosureEvaluation,
    segments: tuple[SequentialClosureSegment, ...],
) -> None:
    if not isinstance(evaluation, RayClosureEvaluation):
        raise FullHandCollisionError(
            "full-hand aggregation needs RayClosureEvaluation"
        )
    candidate = evaluation.candidate
    audit = evaluation.audit
    if candidate is None or audit.failure_reason is not None:
        raise FullHandCollisionError(
            "full-hand aggregation needs a feasible V9 evaluation"
        )
    if (
        audit.method_id != RAY_CLOSURE_METHOD_ID
        or not audit.full_verified_pad_mesh_used
        or audit.pad_face_subset_input_allowed
        or audit.subdivision_budget_exhausted
    ):
        raise FullHandCollisionError(
            "V9 evaluation does not satisfy its registered acceptance contract"
        )
    if len(segments) != 3 or tuple(
        segment.segment_index for segment in segments
    ) != (0, 1, 2):
        raise FullHandCollisionError(
            "full-hand closure path must contain indexed segments 0,1,2"
        )
    pad_names = tuple(segment.pad_name for segment in segments)
    contact_names = tuple(
        row.pad_name for row in candidate.planned_pad_contacts
    )
    if (
        pad_names != audit.pad_order
        or pad_names != contact_names
        or tuple(row.pad_name for row in audit.pad_audits) != pad_names
        or len(audit.independent_actuation_supports) != 3
    ):
        raise FullHandCollisionError(
            "closure segments do not preserve the V9 PAD order"
        )
    joint_names = tuple(backend.hand_model.independent_joint_names)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    closure_open = np.asarray(
        audit.closure_open_joint_positions_rad, dtype=np.float64
    )
    if (
        closure_open.shape != (3,)
        or not np.all(np.isfinite(closure_open))
    ):
        raise FullHandCollisionError(
            "V9 closure-support open values are malformed"
        )
    expected_start: np.ndarray | None = None
    support_union: set[str] = set()
    for index, (segment, pad_audit, support) in enumerate(
        zip(segments, audit.pad_audits, audit.independent_actuation_supports)
    ):
        start = np.asarray(segment.q_start, dtype=np.float64)
        direction = np.asarray(segment.direction, dtype=np.float64)
        if (
            start.shape != (len(joint_names),)
            or direction.shape != (len(joint_names),)
        ):
            raise FullHandCollisionError(
                f"segment {index} dimension does not match the backend"
            )
        if expected_start is not None and not _binary64_array_equal(
            start, expected_start
        ):
            raise FullHandCollisionError(
                f"segment {index} does not start at the previous endpoint"
            )
        if segment.phase.lower != 0.0 or segment.phase.upper <= 0.0:
            raise FullHandCollisionError(
                f"segment {index} must register a positive [0,s] path"
            )
        representative = pad_audit.selected_normalized_closure
        root_lower = pad_audit.selected_root_phase_lower
        root_upper = pad_audit.selected_root_phase_upper
        if (
            representative is None
            or root_lower is None
            or root_upper is None
            or segment.phase.upper != representative
            or not root_lower <= representative <= root_upper
            or pad_audit.first_contact_classification
            != ALLOWED_V9_CONTACT_CLASSIFICATION
            or pad_audit.acceptance_ray_call_count != 0
            or not candidate.planned_pad_contacts[index].surface_coordinates
            or candidate.planned_pad_contacts[index].surface_coordinates[-1]
            != representative
        ):
            raise FullHandCollisionError(
                f"segment {index} is not bound to its V9 transverse root"
            )
        support_names = tuple(str(name) for name in support)
        if (
            not support_names
            or len(set(support_names)) != len(support_names)
            or any(name not in joint_index for name in support_names)
            or support_union.intersection(support_names)
        ):
            raise FullHandCollisionError(
                "V9 actuation supports are absent, unknown, or overlapping"
            )
        support_union.update(support_names)
        observed_support = {
            joint_names[row]
            for row, value in enumerate(direction)
            if value != 0.0
        }
        if observed_support != set(support_names):
            raise FullHandCollisionError(
                f"segment {index} direction differs from its V9 support"
            )
        support_index = joint_index[support_names[0]]
        if not _binary64_array_equal(
            (start[support_index],),
            (closure_open[index],),
        ):
            raise FullHandCollisionError(
                f"segment {index} does not start at its V9 closure-open value"
            )
        pad = backend.hand_model.pads.get(segment.pad_name)
        if pad is None or pad.link_name != segment.active_link_name:
            raise FullHandCollisionError(
                f"segment {index} PAD-to-link binding is not in the hand model"
            )
        endpoint = start + segment.phase.upper * direction
        try:
            backend.hand_model.resolve_joint_positions(start)
            backend.hand_model.resolve_joint_positions(endpoint)
        except ValueError as error:
            raise FullHandCollisionError(
                f"segment {index} leaves the registered joint domain"
            ) from error
        expected_start = endpoint
    final_q = np.asarray(
        candidate.independent_joint_positions_rad, dtype=np.float64
    )
    if final_q.shape != (len(joint_names),) or not _binary64_array_equal(
        final_q, expected_start
    ):
        raise FullHandCollisionError(
            "sequential closure endpoint differs from the V9 candidate"
        )


def _validate_terminal_bindings(
    *,
    link_surfaces: dict[str, HashBoundLinkSurface],
    terminal_surfaces: tuple[TerminalForbiddenSurface, ...],
) -> tuple[dict[str, HashBoundLinkSurface], list[str]]:
    collision_domains = dict(link_surfaces)
    blockers: list[str] = []
    for terminal in terminal_surfaces:
        full_surface = link_surfaces[terminal.link_name]
        partition = terminal.partition
        forbidden_surface = terminal.nonpad_forbidden_surface
        if (
            partition.source_mesh.link_name != terminal.link_name
            or full_surface.source_asset_sha256
            != partition.source_mesh.sha256
            or forbidden_surface.source_asset_sha256
            != partition.source_mesh.sha256
            or partition.source_face_count
            != len(full_surface.triangles_link_m)
        ):
            raise FullHandCollisionError(
                f"terminal source binding mismatch: {terminal.link_name}"
            )
        try:
            source_mesh_data, source_provenance = load_stl_mesh(
                partition.source_mesh.path,
                unit=partition.source_mesh.unit,
                orient_outward=False,
            )
        except (OSError, ValueError) as error:
            raise FullHandCollisionError(
                f"terminal source mesh cannot be deterministically reloaded: "
                f"{terminal.link_name}"
            ) from error
        if source_provenance.source_sha256 != partition.source_mesh.sha256:
            raise FullHandCollisionError(
                f"terminal source provenance changed: {terminal.link_name}"
            )
        source_triangles = np.asarray(
            source_mesh_data.face_vertices_m, dtype=np.float64
        )
        source_keys = triangle_instance_keys(
            source_triangles,
            source_mesh_sha256=partition.source_mesh.sha256,
        )
        partition_keys = partition.pad_allowed + partition.nonpad_forbidden
        if (
            len(set(source_keys)) != len(source_keys)
            or set(source_keys) != set(partition_keys)
        ):
            raise FullHandCollisionError(
                "terminal partition does not cover the full supplied surface: "
                f"{terminal.link_name}"
            )
        source_transform = partition.source_mesh.local_transform
        source_rotation = source_transform[:3, :3]
        source_translation = source_transform[:3, 3]
        derived_full_link = (
            source_triangles @ source_rotation.T + source_translation
        )
        if (
            derived_full_link.shape != full_surface.triangles_link_m.shape
            or _exact_unoriented_triangle_multiset(derived_full_link)
            != _exact_unoriented_triangle_multiset(
                full_surface.triangles_link_m
            )
            or triangle_surface_geometry_sha256(derived_full_link)
            != full_surface.geometry_sha256
        ):
            raise FullHandCollisionError(
                f"full terminal link surface is not the registered source "
                f"unit/transform derivation: {terminal.link_name}"
            )
        forbidden_keys = set(partition.nonpad_forbidden)
        derived_forbidden_source = np.asarray(
            [
                triangle
                for triangle, key in zip(
                    source_triangles, source_keys
                )
                if key in forbidden_keys
            ],
            dtype=np.float64,
        )
        derived_forbidden = (
            derived_forbidden_source @ source_rotation.T
            + source_translation
        )
        if (
            derived_forbidden.shape
            != forbidden_surface.triangles_link_m.shape
            or _exact_unoriented_triangle_multiset(derived_forbidden)
            != _exact_unoriented_triangle_multiset(
                forbidden_surface.triangles_link_m
            )
            or triangle_surface_geometry_sha256(derived_forbidden)
            != forbidden_surface.geometry_sha256
        ):
            raise FullHandCollisionError(
                f"explicit terminal non-PAD surface differs from the exact "
                f"partition: {terminal.link_name}"
            )
        collision_domains[terminal.link_name] = forbidden_surface
        if not partition.formal_collision_eligible:
            blockers.append(
                "TERMINAL_PARTITION_NOT_FORMAL_COLLISION_ELIGIBLE:"
                f"{terminal.link_name}"
            )
    return collision_domains, blockers


def certify_full_hand_sequential_closure(
    *,
    backend: DirectedIntervalKinematics,
    link_surfaces: Sequence[HashBoundLinkSurface],
    terminal_forbidden_surfaces: Sequence[TerminalForbiddenSurface],
    object_surface: HashBoundObjectSurface,
    self_pair_inventory: SelfCollisionPairInventory,
    v9_evaluation: RayClosureEvaluation,
    segments: Sequence[SequentialClosureSegment],
) -> FullHandClosureCollisionCertificate:
    """Aggregate all checkable surface-pair proofs for three closure segments.

    For every segment the function checks every link collision domain against
    the object and every unordered pair in ``self_pair_inventory.all_pairs``.
    Terminal object domains are the exact non-PAD partition only; the omitted
    PAD domain is never silently accepted and creates a mandatory blocker.
    """

    if not isinstance(backend, DirectedIntervalKinematics):
        raise FullHandCollisionError(
            "full-hand aggregation needs DirectedIntervalKinematics"
        )
    if not isinstance(object_surface, HashBoundObjectSurface):
        raise FullHandCollisionError(
            "full-hand object surface is not hash-bound"
        )
    if not isinstance(self_pair_inventory, SelfCollisionPairInventory):
        raise FullHandCollisionError(
            "full-hand self-pair inventory is invalid"
        )
    link_rows = tuple(link_surfaces)
    terminal_rows = tuple(terminal_forbidden_surfaces)
    segment_rows = tuple(segments)
    if not link_rows or not all(
        isinstance(row, HashBoundLinkSurface) for row in link_rows
    ):
        raise FullHandCollisionError(
            "full-hand link surfaces must be explicit and non-empty"
        )
    if not all(
        isinstance(row, TerminalForbiddenSurface) for row in terminal_rows
    ) or not all(
        isinstance(row, SequentialClosureSegment) for row in segment_rows
    ):
        raise FullHandCollisionError(
            "terminal surfaces or closure segments are malformed"
        )
    link_by_name = {row.link_name: row for row in link_rows}
    terminal_by_name = {row.link_name: row for row in terminal_rows}
    if len(link_by_name) != len(link_rows) or len(terminal_by_name) != len(
        terminal_rows
    ):
        raise FullHandCollisionError(
            "link or terminal collision surfaces repeat"
        )
    if tuple(sorted(link_by_name)) != self_pair_inventory.link_names:
        raise FullHandCollisionError(
            "link surfaces do not exactly cover the self-pair inventory"
        )
    active_links = tuple(row.active_link_name for row in segment_rows)
    if (
        len(set(active_links)) != len(active_links)
        or set(active_links) != set(terminal_by_name)
        or not set(terminal_by_name) <= set(link_by_name)
    ):
        raise FullHandCollisionError(
            "terminal inputs must exactly bind the three active PAD links"
        )
    _validate_v9_path_binding(
        backend=backend,
        evaluation=v9_evaluation,
        segments=segment_rows,
    )
    collision_domains, blockers = _validate_terminal_bindings(
        link_surfaces=link_by_name,
        terminal_surfaces=terminal_rows,
    )
    _validate_v9_model_binding(
        backend=backend,
        evaluation=v9_evaluation,
        segments=segment_rows,
        object_surface=object_surface,
        terminal_surfaces=terminal_rows,
    )
    candidate = v9_evaluation.candidate
    if candidate is None:  # narrowed by _validate_v9_path_binding
        raise FullHandCollisionError("V9 candidate unexpectedly disappeared")
    object_from_hand = candidate.object_from_hand_matrix()
    link_object_certificates: list[LinkObjectPathCertificate] = []
    self_pair_certificates: list[SelfPairPathCertificate] = []
    segment_budget_usage: list[tuple[int, int, int]] = []

    for segment in segment_rows:
        remaining_budget = segment.maximum_subdivision_intervals
        for link_name in self_pair_inventory.link_names:
            if remaining_budget == 0:
                blockers.append(
                    "SEGMENT_SHARED_BUDGET_EXHAUSTED_BEFORE_LINK_OBJECT:"
                    f"{segment.segment_index}:{link_name}"
                )
                continue
            surface = collision_domains[link_name]
            child = certify_moving_link_surface_separated_from_static_surface(
                backend=backend,
                link_name=link_name,
                q_start=segment.q_start,
                direction=segment.direction,
                phase=segment.phase,
                object_from_hand_base=object_from_hand,
                moving_triangles_link_m=surface.triangles_link_m,
                static_triangles_object_m=object_surface.triangles_object_m,
                maximum_subdivision_intervals=remaining_budget,
            )
            remaining_budget -= child.audit.processed_interval_count
            if (
                child.audit.moving_surface_geometry_sha256
                != surface.geometry_sha256
                or child.audit.static_surface_geometry_sha256
                != object_surface.geometry_sha256
            ):
                raise FullHandCollisionError(
                    "moving/static child certificate changed a bound surface"
                )
            collision_domain = (
                "TERMINAL_EXACT_NONPAD_FORBIDDEN_SURFACE"
                if link_name in terminal_by_name
                else "FULL_LINK_SURFACE"
            )
            link_object_certificates.append(
                LinkObjectPathCertificate(
                    segment_index=segment.segment_index,
                    link_name=link_name,
                    collision_domain=collision_domain,
                    certificate=child,
                )
            )
            if child.state is not ContinuousCollisionState.CERTIFIED_FREE:
                blockers.append(
                    "LINK_OBJECT_PATH_UNRESOLVED:"
                    f"{segment.segment_index}:{link_name}:"
                    f"{child.audit.unresolved_reason}"
                )
        for first_link, second_link in self_pair_inventory.all_pairs:
            if remaining_budget == 0:
                blockers.append(
                    "SEGMENT_SHARED_BUDGET_EXHAUSTED_BEFORE_SELF_PAIR:"
                    f"{segment.segment_index}:{first_link}:{second_link}"
                )
                continue
            first_surface = link_by_name[first_link]
            second_surface = link_by_name[second_link]
            child_pair = (
                certify_moving_link_surfaces_separated_from_each_other(
                    backend=backend,
                    first_link_name=first_link,
                    second_link_name=second_link,
                    q_start=segment.q_start,
                    direction=segment.direction,
                    phase=segment.phase,
                    object_from_hand_base=object_from_hand,
                    first_triangles_link_m=(
                        first_surface.triangles_link_m
                    ),
                    second_triangles_link_m=(
                        second_surface.triangles_link_m
                    ),
                    maximum_subdivision_intervals=remaining_budget,
                )
            )
            remaining_budget -= child_pair.audit.processed_interval_count
            if (
                child_pair.audit.first_surface_geometry_sha256
                != first_surface.geometry_sha256
                or child_pair.audit.second_surface_geometry_sha256
                != second_surface.geometry_sha256
            ):
                raise FullHandCollisionError(
                    "moving/moving child certificate changed a bound surface"
                )
            self_pair_certificates.append(
                SelfPairPathCertificate(
                    segment_index=segment.segment_index,
                    first_link_name=first_link,
                    second_link_name=second_link,
                    certificate=child_pair,
                )
            )
            if child_pair.state is not ContinuousCollisionState.CERTIFIED_FREE:
                blockers.append(
                    "SELF_PAIR_PATH_UNRESOLVED:"
                    f"{segment.segment_index}:{first_link}:{second_link}:"
                    f"{child_pair.audit.unresolved_reason}"
                )
        segment_budget_usage.append(
            (
                segment.maximum_subdivision_intervals,
                segment.maximum_subdivision_intervals - remaining_budget,
                remaining_budget,
            )
        )

    for segment in segment_rows:
        blockers.append(
            f"{PAD_SURFACE_BLOCKER_PREFIX}:"
            f"{segment.pad_name}:{segment.active_link_name}"
        )
    blockers.extend(
        (
            "V9_REPRESENTATIVE_PHASE_IS_ROOT_BRACKET_MIDPOINT_"
            "NOT_EXACT_CONTACT_ENDPOINT",
            "SOLID_CONTAINMENT_OR_INITIAL_OUTSIDE_CERTIFICATE_UNAVAILABLE",
            "AUTHORITATIVE_FULL_HAND_COLLISION_LINK_ROSTER_NOT_PROVEN",
        )
    )
    expected_link_object = len(segment_rows) * len(link_rows)
    expected_self_pairs = (
        len(segment_rows) * len(self_pair_inventory.all_pairs)
    )
    free_link_object = sum(
        row.certificate.state is ContinuousCollisionState.CERTIFIED_FREE
        for row in link_object_certificates
    )
    free_self_pairs = sum(
        row.certificate.state is ContinuousCollisionState.CERTIFIED_FREE
        for row in self_pair_certificates
    )
    ordered_link_bindings = tuple(
        (
            name,
            link_by_name[name].source_asset_sha256,
            link_by_name[name].geometry_sha256,
        )
        for name in self_pair_inventory.link_names
    )
    ordered_terminal_bindings = tuple(
        (
            name,
            terminal_by_name[name].partition.partition_sha256,
            terminal_by_name[name].nonpad_forbidden_surface.geometry_sha256,
        )
        for name in sorted(terminal_by_name)
    )
    audit = FullHandClosureCollisionAudit(
        method_id=METHOD_ID,
        interval_kinematics_method_id=INTERVAL_KINEMATICS_METHOD_ID,
        ray_closure_method_id=RAY_CLOSURE_METHOD_ID,
        v9_evidence_sha256=_v9_evidence_sha256(v9_evaluation),
        object_id=object_surface.object_id,
        object_source_asset_sha256=object_surface.source_asset_sha256,
        object_surface_geometry_sha256=object_surface.geometry_sha256,
        ray_closure_object_geometry_sha256=(
            object_surface.ray_closure_object_geometry_sha256
        ),
        ray_model_contract_sha256=(
            v9_evaluation.audit.model_contract_sha256
        ),
        link_surface_bindings=ordered_link_bindings,
        terminal_partition_bindings=ordered_terminal_bindings,
        self_pair_inventory_sha256=self_pair_inventory.inventory_sha256,
        segment_contract_sha256=tuple(
            _segment_sha256(row) for row in segment_rows
        ),
        segment_budget_usage=tuple(segment_budget_usage),
        segment_count=len(segment_rows),
        link_count=len(link_rows),
        terminal_link_count=len(terminal_rows),
        self_pair_count_per_segment=len(self_pair_inventory.all_pairs),
        expected_link_object_domain_count=expected_link_object,
        evaluated_link_object_domain_count=len(link_object_certificates),
        certified_free_link_object_domain_count=free_link_object,
        expected_self_pair_domain_count=expected_self_pairs,
        evaluated_self_pair_domain_count=len(self_pair_certificates),
        certified_free_self_pair_domain_count=free_self_pairs,
        all_link_object_domains_covered=(
            len(link_object_certificates) == expected_link_object
        ),
        all_self_pair_domains_covered=(
            len(self_pair_certificates) == expected_self_pairs
        ),
        srdf_exemptions_applied=False,
        pad_endpoint_continuous_surface_certificate_present=False,
        checkable_collision_gates_passed=(
            free_link_object == expected_link_object
            and free_self_pairs == expected_self_pairs
        ),
        blockers=tuple(sorted(set(blockers))),
        claim_limitations=CLAIM_LIMITATIONS,
    )
    return FullHandClosureCollisionCertificate(
        state=FullHandClosureCollisionState.NOT_CERTIFIABLE,
        link_object_certificates=tuple(link_object_certificates),
        self_pair_certificates=tuple(self_pair_certificates),
        audit=audit,
    )


__all__ = [
    "ALLOWED_V9_CONTACT_CLASSIFICATION",
    "CLAIM_LIMITATIONS",
    "FullHandClosureCollisionAudit",
    "FullHandClosureCollisionCertificate",
    "FullHandClosureCollisionState",
    "FullHandCollisionError",
    "HashBoundLinkSurface",
    "HashBoundObjectSurface",
    "LinkObjectPathCertificate",
    "METHOD_ID",
    "PAD_SURFACE_BLOCKER_PREFIX",
    "SURFACE_HASH_METHOD_ID",
    "SelfPairPathCertificate",
    "SequentialClosureSegment",
    "TerminalForbiddenSurface",
    "certify_full_hand_sequential_closure",
    "triangle_surface_geometry_sha256",
]
