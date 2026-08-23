"""Cheap, claim-limited ordering before expensive V9 evaluation.

The proxy consumes only the conservative whole-path screen audit that has
already been computed for every canonical parameter row.  It may reorder
survivors, but it cannot certify contact, collision freedom, grasp quality, or
dynamic success.  The hard Top-K bound is therefore a compute-budget policy,
not a geometric proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .ray_closure import WholePathPadSphereScreen


METHOD_ID = "CARTS_THREE_PAD_CHEAP_SCREEN_LEXICOGRAPHIC_PROXY_RANK_V1"
EVIDENCE_ROLE = "ORDER_ONLY_NEVER_CERTIFIES_FREE_CONTACT_OR_SUCCESS"
EXACT_TOP_K = 4
PARAMETER_DIMENSION = 5
ROOT_SEGMENT_COUNT = 8


class MultifidelityCandidateRankError(ValueError):
    """Raised when proxy inputs could be mistaken for formal evidence."""


@dataclass(frozen=True)
class ProxyRankInput:
    """One canonical candidate and its already-computed cheap screen."""

    canonical_key_hex: str
    parameters_unit: tuple[float, float, float, float, float]
    first_attempt_index: int
    pad_screens: tuple[WholePathPadSphereScreen, ...]

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters_unit, dtype=np.float64)
        if (
            parameters.shape != (PARAMETER_DIMENSION,)
            or not np.all(np.isfinite(parameters))
            or np.any(parameters < 0.0)
            or np.any(parameters > 1.0)
            or parameters[0] >= 1.0
        ):
            raise MultifidelityCandidateRankError(
                "proxy parameters are not canonical five-dimensional V9 values"
            )
        try:
            raw_key = bytes.fromhex(self.canonical_key_hex)
        except ValueError as error:
            raise MultifidelityCandidateRankError(
                "proxy canonical key is not hexadecimal"
            ) from error
        expected_key = np.asarray(parameters, dtype=">f8").tobytes(order="C")
        if len(raw_key) != 8 * PARAMETER_DIMENSION or raw_key != expected_key:
            raise MultifidelityCandidateRankError(
                "proxy canonical key differs from its binary64 parameters"
            )
        if (
            isinstance(self.first_attempt_index, bool)
            or not isinstance(self.first_attempt_index, int)
            or self.first_attempt_index < 0
        ):
            raise MultifidelityCandidateRankError(
                "proxy first attempt index must be a nonnegative integer"
            )
        if (
            len(self.pad_screens) != 3
            or len({row.pad_name for row in self.pad_screens}) != 3
        ):
            raise MultifidelityCandidateRankError(
                "proxy requires exactly three distinct PAD screen rows"
            )
        for screen in self.pad_screens:
            if type(screen) is not WholePathPadSphereScreen:
                raise MultifidelityCandidateRankError(
                    "proxy PAD screen has an unexpected type"
                )
            if not 0 <= screen.root_overlap_segment_count <= ROOT_SEGMENT_COUNT:
                raise MultifidelityCandidateRankError(
                    "proxy root overlap count lies outside the eight path segments"
                )

    @property
    def cheap_rejected(self) -> bool:
        return any(
            row.certified_free or row.certified_no_valid_contact
            for row in self.pad_screens
        )


@dataclass(frozen=True)
class ProxyRankedCandidate:
    """Auditable order-only metrics for one cheap-screen survivor."""

    canonical_key_hex: str
    parameters_unit: tuple[float, float, float, float, float]
    first_attempt_index: int
    proxy_rank: int
    selected_for_exact_v9: bool
    minimum_root_overlap_segment_count: int
    root_overlap_segment_count_imbalance: int
    total_root_overlap_segment_count: int
    total_obb_sat_triangle_test_count: int
    total_spatial_node_query_count: int
    minimum_nonperiodic_unit_boundary_margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_key_hex": self.canonical_key_hex,
            "parameters_unit": list(self.parameters_unit),
            "first_attempt_index": self.first_attempt_index,
            "proxy_rank": self.proxy_rank,
            "selected_for_exact_v9": self.selected_for_exact_v9,
            "minimum_root_overlap_segment_count": (
                self.minimum_root_overlap_segment_count
            ),
            "root_overlap_segment_count_imbalance": (
                self.root_overlap_segment_count_imbalance
            ),
            "total_root_overlap_segment_count": (
                self.total_root_overlap_segment_count
            ),
            "total_obb_sat_triangle_test_count": (
                self.total_obb_sat_triangle_test_count
            ),
            "total_spatial_node_query_count": (
                self.total_spatial_node_query_count
            ),
            "minimum_nonperiodic_unit_boundary_margin": (
                self.minimum_nonperiodic_unit_boundary_margin
            ),
            "evidence_role": EVIDENCE_ROLE,
            "proxy_certifies_or_rejects": False,
        }


@dataclass(frozen=True)
class MultifidelityProxyRankResult:
    """Complete cheap-screen partition and the fixed exact-work ceiling."""

    input_count: int
    cheap_rejected_keys: tuple[str, ...]
    ranked_survivors: tuple[ProxyRankedCandidate, ...]
    exact_selected_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.exact_selected_keys) > EXACT_TOP_K:
            raise MultifidelityCandidateRankError(
                "proxy result exceeds the frozen exact Top-K ceiling"
            )
        selected_from_rows = tuple(
            row.canonical_key_hex
            for row in self.ranked_survivors
            if row.selected_for_exact_v9
        )
        if selected_from_rows != self.exact_selected_keys:
            raise MultifidelityCandidateRankError(
                "proxy selected-key list differs from ranked rows"
            )
        if self.input_count != (
            len(self.cheap_rejected_keys) + len(self.ranked_survivors)
        ):
            raise MultifidelityCandidateRankError(
                "proxy result does not partition every unique input"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": METHOD_ID,
            "evidence_role": EVIDENCE_ROLE,
            "proxy_certifies_or_rejects": False,
            "input_unique_count": self.input_count,
            "cheap_rejected_unique_count": len(self.cheap_rejected_keys),
            "cheap_rejected_keys": list(self.cheap_rejected_keys),
            "survivor_unique_count": len(self.ranked_survivors),
            "exact_top_k_ceiling": EXACT_TOP_K,
            "exact_selected_count": len(self.exact_selected_keys),
            "exact_selected_keys": list(self.exact_selected_keys),
            "ranked_survivors": [
                row.as_dict() for row in self.ranked_survivors
            ],
            "formal_selected_contact_range_policy": None,
            "formal_selected_candidate": None,
            "full_hand_collision_state": "NOT_CERTIFIABLE",
            "dynamic_launch_allowed": False,
            "hardware_authorized": False,
        }


def _ranked_candidate(row: ProxyRankInput) -> ProxyRankedCandidate:
    if row.cheap_rejected:
        raise MultifidelityCandidateRankError(
            "a cheap-rejected candidate cannot enter proxy ranking"
        )
    if any(screen.skipped_due_to_other_pad_free for screen in row.pad_screens):
        raise MultifidelityCandidateRankError(
            "a survivor proxy requires complete audits for all three PADs"
        )
    root_counts = tuple(
        int(screen.root_overlap_segment_count) for screen in row.pad_screens
    )
    nonperiodic = row.parameters_unit[1:]
    boundary_margin = min(min(value, 1.0 - value) for value in nonperiodic)
    if not math.isfinite(boundary_margin):  # pragma: no cover - input guard
        raise MultifidelityCandidateRankError(
            "proxy boundary margin is not finite"
        )
    return ProxyRankedCandidate(
        canonical_key_hex=row.canonical_key_hex,
        parameters_unit=row.parameters_unit,
        first_attempt_index=row.first_attempt_index,
        proxy_rank=0,
        selected_for_exact_v9=False,
        minimum_root_overlap_segment_count=min(root_counts),
        root_overlap_segment_count_imbalance=(
            max(root_counts) - min(root_counts)
        ),
        total_root_overlap_segment_count=sum(root_counts),
        total_obb_sat_triangle_test_count=sum(
            int(screen.obb_sat_triangle_test_count)
            for screen in row.pad_screens
        ),
        total_spatial_node_query_count=sum(
            int(screen.spatial_node_query_count)
            for screen in row.pad_screens
        ),
        minimum_nonperiodic_unit_boundary_margin=float(boundary_margin),
    )


def _order_key(row: ProxyRankedCandidate) -> tuple[object, ...]:
    """Frozen no-weight lexicographic order; lower tuple sorts first."""

    return (
        -row.minimum_root_overlap_segment_count,
        row.root_overlap_segment_count_imbalance,
        -row.total_root_overlap_segment_count,
        row.total_obb_sat_triangle_test_count,
        row.total_spatial_node_query_count,
        -row.minimum_nonperiodic_unit_boundary_margin,
        row.canonical_key_hex,
    )


def rank_screened_candidates(
    rows: Sequence[ProxyRankInput],
) -> MultifidelityProxyRankResult:
    """Partition cheap rejections and select at most four ordered survivors."""

    inputs = tuple(rows)
    if not inputs:
        raise MultifidelityCandidateRankError(
            "proxy ranking requires at least one canonical input"
        )
    keys = tuple(row.canonical_key_hex for row in inputs)
    if len(set(keys)) != len(keys):
        raise MultifidelityCandidateRankError(
            "proxy ranking inputs must already be canonical and unique"
        )
    rejected_keys = tuple(
        sorted(row.canonical_key_hex for row in inputs if row.cheap_rejected)
    )
    ordered = sorted(
        (_ranked_candidate(row) for row in inputs if not row.cheap_rejected),
        key=_order_key,
    )
    ranked = tuple(
        replace(
            row,
            proxy_rank=index + 1,
            selected_for_exact_v9=index < EXACT_TOP_K,
        )
        for index, row in enumerate(ordered)
    )
    selected = tuple(
        row.canonical_key_hex
        for row in ranked[:EXACT_TOP_K]
    )
    return MultifidelityProxyRankResult(
        input_count=len(inputs),
        cheap_rejected_keys=rejected_keys,
        ranked_survivors=ranked,
        exact_selected_keys=selected,
    )


__all__ = [
    "EVIDENCE_ROLE",
    "EXACT_TOP_K",
    "METHOD_ID",
    "MultifidelityCandidateRankError",
    "MultifidelityProxyRankResult",
    "ProxyRankInput",
    "ProxyRankedCandidate",
    "rank_screened_candidates",
]
