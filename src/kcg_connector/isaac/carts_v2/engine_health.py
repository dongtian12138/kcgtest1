#!/usr/bin/env python3

"""Small fail-closed boundary for PhysX-backed V2 dynamic evidence."""

from __future__ import annotations

import re
from typing import Mapping


ENGINE_EVIDENCE_FIELDS = (
    "controller_preflight_pass",
    "engine_health_pass",
    "accepted_preflight_pass",
    "physx_capacity_warning_count",
    "configured_gpu_found_lost_aggregate_pairs_capacity",
    "configured_gpu_total_aggregate_pairs_capacity",
    "observed_gpu_found_lost_aggregate_pairs_peak",
    "observed_gpu_total_aggregate_pairs_peak",
    "engine_log_sha256",
    "identity_hash_check_pass",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def pending_engine_fields(
    controller_preflight_pass: bool, identity_hash_check_pass: bool
) -> dict[str, object]:
    """Return the complete schema while engine evidence is still unavailable."""

    return {
        "controller_preflight_pass": bool(controller_preflight_pass),
        "engine_health_pass": False,
        "accepted_preflight_pass": False,
        "physx_capacity_warning_count": None,
        "configured_gpu_found_lost_aggregate_pairs_capacity": None,
        "configured_gpu_total_aggregate_pairs_capacity": None,
        "observed_gpu_found_lost_aggregate_pairs_peak": None,
        "observed_gpu_total_aggregate_pairs_peak": None,
        "engine_log_sha256": None,
        "identity_hash_check_pass": bool(identity_hash_check_pass),
    }


def preflight_is_accepted(document: Mapping[str, object]) -> bool:
    """Require the complete engine-backed preflight record; missing means reject."""

    if any(field not in document for field in ENGINE_EVIDENCE_FIELDS):
        return False
    controller = document.get("controller_preflight_pass") is True
    engine = document.get("engine_health_pass") is True
    identity = document.get("identity_hash_check_pass") is True
    accepted = document.get("accepted_preflight_pass") is True
    try:
        found_capacity = int(
            document["configured_gpu_found_lost_aggregate_pairs_capacity"]
        )
        total_capacity = int(document["configured_gpu_total_aggregate_pairs_capacity"])
        found_peak = int(document["observed_gpu_found_lost_aggregate_pairs_peak"])
        total_peak = int(document["observed_gpu_total_aggregate_pairs_peak"])
    except (TypeError, ValueError):
        return False
    log_sha = document.get("engine_log_sha256")
    capacity_evidence = bool(
        document.get("physx_capacity_warning_count") == 0
        and 0 <= found_peak < found_capacity
        and 0 <= total_peak < total_capacity
        and isinstance(log_sha, str)
        and _SHA256.fullmatch(log_sha)
    )
    expected = controller and engine and identity and capacity_evidence
    return accepted and expected


__all__ = [
    "ENGINE_EVIDENCE_FIELDS", "pending_engine_fields", "preflight_is_accepted"
]
