#!/usr/bin/env python3

"""Fail-closed contract for an opt-in render-only tooth ghost run.

The prepared twist probe already proves that all 24 tooth children remain
rigid with the CouplingNut parent, but three opaque fingers hide eight tooth
IDs from the synchronized RGB audit.  The smallest useful follow-up is a new
run with *only* inherited USD ``visibility`` opinions authored on the three
finger roots in the anonymous session layer.  USD visibility is a
presentation property; the finger rigid bodies and colliders remain in the
same contact simulation.

This module deliberately contains no Isaac Sim imports.  It validates the
runtime sidecar, exact capture reuse, physics/contact equivalence and visual
coverage after a GPU run.  Runtime USD authoring belongs in a tiny injected
adapter, not in this CPU evidence contract or the stable end-to-end runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "kcg_d38999_tooth_occlusion_control_v1"
RUNTIME_SCHEMA_VERSION = "kcg_d38999_tooth_ghost_runtime_v1"
CAPTURE_SCHEMA_VERSION = "kcg_d38999_tooth_sync_capture_v3"
PHYSICS_SCHEMA_VERSION = "kcg_d38999_nut_tooth_jitter_probe_v1"
EXPECTED_SEGMENTS = tuple(f"Segment_{index:02d}" for index in range(24))
EXPECTED_SEGMENT_SET = frozenset(EXPECTED_SEGMENTS)
EXPECTED_FINGER_ROOT_SUFFIXES = (
    "/f1Link1",
    "/f2Link1",
    "/f3Link1",
)
EXPECTED_ROBOT_ROOT = "/World/HandArm"
EXPECTED_VIEWS = (
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
)


class OcclusionControlError(RuntimeError):
    """Raised when a ghost-run claim is not provenance-safe."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OcclusionControlError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OcclusionControlError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OcclusionControlError(f"{label} must be finite")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OcclusionControlError(
            f"{label} must be a non-negative integer"
        )
    return value


def _sha256_document(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def contact_dynamics_fingerprint(report: Mapping[str, Any]) -> str:
    """Hash the physical/contact fields that a render ghost must preserve.

    The ordinary physics trace intentionally omits verbose contact records.
    This second fingerprint prevents an apparently identical parent motion
    from hiding a changed finger/tooth or entry-shell contact solution.
    Output paths and presentation-only color metadata are intentionally not
    included because they differ between evidence directories.
    """

    report = _mapping(report, "physics report")
    if report.get("schema_version") != PHYSICS_SCHEMA_VERSION:
        raise OcclusionControlError("physics report schema mismatch")
    segments = _mapping(report.get("segment_aggregate"), "segments")
    if set(segments) != EXPECTED_SEGMENT_SET:
        raise OcclusionControlError(
            "physics report must contain exactly 24 tooth segments"
        )
    segment_payload = {}
    for name in EXPECTED_SEGMENTS:
        item = _mapping(segments[name], name)
        counterparts = item.get("contact_counterparts")
        if not isinstance(counterparts, list) or not all(
            isinstance(path, str) for path in counterparts
        ):
            raise OcclusionControlError(
                f"{name}.contact_counterparts must be a string list"
            )
        minimum_separation = item.get("minimum_contact_separation_m")
        if minimum_separation is not None:
            minimum_separation = _finite(
                minimum_separation,
                f"{name}.minimum_contact_separation_m",
            )
        segment_payload[name] = {
            "contact_counterparts": sorted(counterparts),
            "contact_records": _nonnegative_integer(
                item.get("contact_records"), f"{name}.contact_records"
            ),
            "maximum_contact_impulse_norm": _finite(
                item.get("maximum_contact_impulse_norm"),
                f"{name}.maximum_contact_impulse_norm",
            ),
            "minimum_contact_separation_m": minimum_separation,
        }
        for field in (
            "maximum_local_rotation_error_rad",
            "maximum_local_translation_error_m",
            "maximum_parent_relative_rotation_error_rad",
            "maximum_parent_relative_translation_error_m",
        ):
            segment_payload[name][field] = _finite(
                item.get(field), f"{name}.{field}"
            )
    payload = {
        "anomaly_steps": _nonnegative_integer(
            report.get("anomaly_steps"), "anomaly_steps"
        ),
        "phase_steps": dict(
            sorted(_mapping(report.get("phase_steps"), "phase_steps").items())
        ),
        "segments": segment_payload,
        "steps": _nonnegative_integer(report.get("steps"), "steps"),
        "thresholds": dict(
            sorted(_mapping(report.get("thresholds"), "thresholds").items())
        ),
    }
    return _sha256_document(payload)


def validate_runtime_sidecar(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Require visibility-only session authoring before physics starts."""

    sidecar = _mapping(sidecar, "ghost runtime sidecar")
    if sidecar.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise OcclusionControlError("ghost runtime schema mismatch")
    if sidecar.get("passed") is not True:
        raise OcclusionControlError("ghost runtime did not pass")
    authoring = _mapping(sidecar.get("authoring"), "authoring")
    if authoring.get("edit_layer") != "anonymous_session_layer":
        raise OcclusionControlError("ghost edit was not session-layer only")
    if authoring.get("robot_root") != EXPECTED_ROBOT_ROOT:
        raise OcclusionControlError("ghost robot root differs")
    paths = authoring.get("prim_paths")
    if not isinstance(paths, list) or len(paths) != 3:
        raise OcclusionControlError("ghost must target exactly three fingers")
    if any(not path.startswith(EXPECTED_ROBOT_ROOT + "/") for path in paths):
        raise OcclusionControlError("ghost target escapes the robot root")
    # Compare with endswith instead of depending on the imported robot's long
    # nested link chain.  The exact resolved paths remain recorded in evidence.
    matched_suffixes = tuple(
        suffix
        for suffix in EXPECTED_FINGER_ROOT_SUFFIXES
        if sum(path.endswith(suffix) for path in paths) == 1
    )
    if matched_suffixes != EXPECTED_FINGER_ROOT_SUFFIXES:
        raise OcclusionControlError("ghost finger-root identities differ")
    if authoring.get("properties") != ["visibility"]:
        raise OcclusionControlError("ghost authored a non-visibility property")
    for key in (
        "authored_before_timeline_play",
        "authored_before_world_reset",
        "computed_descendants_invisible",
    ):
        if authoring.get(key) is not True:
            raise OcclusionControlError(f"ghost authoring gate failed: {key}")
    if authoring.get("visibility_token") != "invisible":
        raise OcclusionControlError("ghost visibility token differs")
    mutations = _mapping(sidecar.get("mutation_audit"), "mutation_audit")
    expected_zero = (
        "collision_api_writes",
        "material_writes",
        "object_pose_writes",
        "physics_api_writes",
        "xform_writes",
    )
    for key in expected_zero:
        if mutations.get(key) != 0:
            raise OcclusionControlError(f"ghost mutation gate failed: {key}")
    cleanup = _mapping(sidecar.get("cleanup"), "cleanup")
    if (
        cleanup.get("session_visibility_opinions_removed") is not True
        or cleanup.get("effective_visibility_restored_to_pre_author_state")
        is not True
    ):
        raise OcclusionControlError("ghost visibility cleanup failed")
    source = _mapping(sidecar.get("source"), "source")
    if (
        source.get("unchanged_during_run") is not True
        or source.get("runner_unchanged_during_run") is not True
    ):
        raise OcclusionControlError("ghost adapter source changed during run")
    runtime_hashes = {
        source.get("sha256_at_import"),
        source.get("sha256_at_start"),
        source.get("sha256_at_finalize"),
    }
    runner_hashes = {
        source.get("runner_sha256_at_start"),
        source.get("runner_sha256_at_finalize"),
    }
    if (
        None in runtime_hashes
        or len(runtime_hashes) != 1
        or None in runner_hashes
        or len(runner_hashes) != 1
    ):
        raise OcclusionControlError("ghost runtime provenance hashes differ")
    return {
        "finger_root_paths": list(paths),
        "session_layer_only": True,
        "visibility_only": True,
        "zero_physics_or_pose_writes": True,
    }


def _validate_capture_pair(
    baseline: Mapping[str, Any], ghost: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = _mapping(baseline, "baseline capture manifest")
    ghost = _mapping(ghost, "ghost capture manifest")
    for label, manifest in (("baseline", baseline), ("ghost", ghost)):
        if manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION:
            raise OcclusionControlError(f"{label} capture schema mismatch")
        if manifest.get("passed") is not True:
            raise OcclusionControlError(f"{label} capture did not pass")
    for key in (
        "camera_rig",
        "frame_capture",
        "render_settings",
        "sampling",
        "step_mapping_semantics",
        "sync_columns",
    ):
        if baseline.get(key) != ghost.get(key):
            raise OcclusionControlError(f"capture {key} changed")
    rig = _mapping(ghost.get("camera_rig"), "camera rig")
    if tuple(rig.get("view_priority", ())) != EXPECTED_VIEWS:
        raise OcclusionControlError("ghost did not reuse the four-view rig")
    helper_hashes = []
    helper_paths = []
    for label, manifest in (("baseline", baseline), ("ghost", ghost)):
        source = _mapping(
            manifest.get("capture_source"), f"{label} capture source"
        )
        if source.get("unchanged_during_capture") is not True:
            raise OcclusionControlError(
                f"{label} capture helper changed during run"
            )
        hashes = {
            source.get("sha256_at_import"),
            source.get("sha256_at_start"),
            source.get("sha256_at_finalize"),
        }
        if None in hashes or len(hashes) != 1:
            raise OcclusionControlError(
                f"{label} capture helper provenance hashes differ"
            )
        helper_hashes.append(next(iter(hashes)))
        helper_paths.append(source.get("path"))
        cleanup = _mapping(
            manifest.get("cleanup"), f"{label} capture cleanup"
        )
        if cleanup.get("object_pose_writes") != 0:
            raise OcclusionControlError(
                f"{label} capture helper wrote an object pose"
            )
    if len(set(helper_hashes)) != 1 or len(set(helper_paths)) != 1:
        raise OcclusionControlError(
            "ghost capture helper differs from baseline"
        )
    return {
        "camera_rig_identical": True,
        "capture_helper_unchanged": True,
        "same_completed_physics_step_contract": bool(
            rig.get("same_completed_physics_step") is True
        ),
    }


def validate_strict_coverage(coverage: Mapping[str, Any]) -> dict[str, Any]:
    """Require all 24 identities in the union at every sampled transition."""

    coverage = _mapping(coverage, "visual coverage")
    observed = set(coverage.get("segments_in_identity_union", ()))
    missing = coverage.get("missing_from_identity_union")
    if observed != EXPECTED_SEGMENT_SET or missing != []:
        raise OcclusionControlError("ghost visual identity union is not 24/24")
    if coverage.get("identity_union_all_24") is not True:
        raise OcclusionControlError("ghost 24-ID union gate failed")
    if coverage.get("every_transition_all_24") is not True:
        raise OcclusionControlError(
            "ghost does not expose all 24 IDs at every transition"
        )
    counts = _mapping(
        coverage.get("per_segment_transition_counts"),
        "per-segment transition counts",
    )
    if set(counts) != EXPECTED_SEGMENT_SET or min(counts.values()) <= 0:
        raise OcclusionControlError("a tooth has no measurable transition")
    per_phase = _mapping(coverage.get("per_phase"), "per-phase coverage")
    for phase, item in per_phase.items():
        item = _mapping(item, f"coverage.{phase}")
        if (
            item.get("identity_union_all_24") is not True
            or item.get("missing_from_identity_union") != []
        ):
            raise OcclusionControlError(
                f"phase {phase} does not cover all 24 tooth IDs"
            )
    return {
        "every_sampled_transition_all_24": True,
        "identity_union_all_24": True,
        "minimum_segment_transition_count": min(counts.values()),
    }


def evaluate_occlusion_control(
    *,
    baseline_capture_manifest: Mapping[str, Any],
    ghost_capture_manifest: Mapping[str, Any],
    baseline_physics_report: Mapping[str, Any],
    ghost_physics_report: Mapping[str, Any],
    baseline_physics_trace_sha256: str,
    ghost_physics_trace_sha256: str,
    baseline_contact_trace_sha256: str,
    ghost_contact_trace_sha256: str,
    ghost_runtime_sidecar: Mapping[str, Any],
    visual_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize inspection only after render, physics and coverage gates."""

    if (
        not isinstance(baseline_physics_trace_sha256, str)
        or len(baseline_physics_trace_sha256) != 64
        or baseline_physics_trace_sha256
        != ghost_physics_trace_sha256
    ):
        raise OcclusionControlError("ghost physics state trace differs")
    if (
        not isinstance(baseline_contact_trace_sha256, str)
        or len(baseline_contact_trace_sha256) != 64
        or baseline_contact_trace_sha256 != ghost_contact_trace_sha256
    ):
        raise OcclusionControlError("ghost per-step contact trace differs")
    runtime = validate_runtime_sidecar(ghost_runtime_sidecar)
    capture = _validate_capture_pair(
        baseline_capture_manifest, ghost_capture_manifest
    )
    baseline_contact = contact_dynamics_fingerprint(baseline_physics_report)
    ghost_contact = contact_dynamics_fingerprint(ghost_physics_report)
    if baseline_contact != ghost_contact:
        raise OcclusionControlError("ghost contact dynamics differ")
    coverage = validate_strict_coverage(visual_coverage)
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "VALID_OCCLUSION_CONTROL_READY_FOR_PIXEL_INSPECTION",
        "passed": True,
        "capture": capture,
        "contact_dynamics_sha256": ghost_contact,
        "coverage": coverage,
        "physics": {
            "contact_dynamics_identical": True,
            "contact_trace_identical": True,
            "contact_trace_sha256": ghost_contact_trace_sha256,
            "state_trace_identical": True,
            "trace_sha256": ghost_physics_trace_sha256,
        },
        "runtime": runtime,
        "claim_boundaries": {
            "no_render_jitter_absence_claim_without_registered_threshold": (
                True
            ),
            "presentation_only_diagnostic": True,
            "safety_thresholds_changed": False,
        },
    }


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "EXPECTED_FINGER_ROOT_SUFFIXES",
    "EXPECTED_SEGMENTS",
    "OcclusionControlError",
    "PHYSICS_SCHEMA_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "contact_dynamics_fingerprint",
    "evaluate_occlusion_control",
    "validate_runtime_sidecar",
    "validate_strict_coverage",
]
