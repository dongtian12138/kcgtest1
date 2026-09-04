#!/usr/bin/env python3

"""Small fail-closed boundary for PhysX-backed V2 dynamic evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping


ENGINE_EVIDENCE_FIELDS = (
    "controller_preflight_pass", "engine_health_pass", "accepted_preflight_pass",
    "physx_capacity_warning_count",
    "configured_gpu_found_lost_aggregate_pairs_capacity",
    "configured_gpu_total_aggregate_pairs_capacity",
    "observed_gpu_found_lost_aggregate_pairs_peak",
    "observed_gpu_total_aggregate_pairs_peak",
    "engine_log_sha256", "engine_log_sync_marker", "engine_log_marker_seen",
    "engine_log_audit_byte_count", "engine_log_audit_boundary",
    "physx_error_lines", "identity_hash_check_pass",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPACITY_REQUEST = re.compile(
    r"PxGpuDynamicsMemoryConfig::(foundLostAggregatePairsCapacity|"
    r"totalAggregatePairsCapacity)\s+to\s+(\d+)", re.IGNORECASE,
)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def load_runtime_resources(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("hardware_authorized") is not False
            or document.get("algorithm_or_physics_parameter") is not False):
        raise ValueError("runtime resources changed authorization or physics identity")
    if not _SHA256.fullmatch(str(document.get("capacity_audit_sha256"))):
        raise ValueError("runtime resources do not bind the capacity audit")
    if document.get("capacity_rule") != {
        "multiplier": 2.0, "rounding": "NEXT_POWER_OF_TWO"
    }:
        raise ValueError("GPU capacity rule is not the frozen 2x power-of-two rule")
    pairs = (
        ("gpu_found_lost_aggregate_pairs_capacity", "observed_found_lost_peak"),
        ("gpu_total_aggregate_pairs_capacity", "observed_total_peak"),
    )
    for capacity_key, peak_key in pairs:
        peak = int(document["capacity_basis"][peak_key])
        if int(document[capacity_key]) != _next_power_of_two(2 * peak):
            raise ValueError(f"{capacity_key} differs from the frozen resource rule")
    return document


def gpu_world_parameters(resources) -> dict[str, object]:
    return {
        "backend": "numpy", "device": "cuda:0",
        "sim_params": {
            "use_gpu_pipeline": True,
            "gpu_found_lost_aggregate_pairs_capacity": int(resources[
                "gpu_found_lost_aggregate_pairs_capacity"]),
            "gpu_total_aggregate_pairs_capacity": int(resources[
                "gpu_total_aggregate_pairs_capacity"]),
        },
    }


def gpu_backend_record(world, context) -> dict[str, object]:
    result = {
        "requested_device": "cuda:0", "actual_data_backend": str(world.backend),
        "world_device": str(world.device), "physics_context_device": str(context.device),
        "gpu_sim": bool(context.use_gpu_sim), "gpu_pipeline": bool(context.use_gpu_pipeline),
        "gpu_dynamics_enabled": bool(context.is_gpu_dynamics_enabled()),
        "broadphase_type": str(context.get_broadphase_type()),
    }
    result["pass"] = bool(
        "cuda" in result["world_device"] and result["gpu_sim"] and result["gpu_pipeline"]
        and result["gpu_dynamics_enabled"]
        and result["broadphase_type"] == "GPU"
    )
    return result


def current_engine_log_path() -> Path:
    import carb

    return Path(carb.tokens.get_tokens_interface().resolve("${log_file}")).resolve()


class PhysxStatsMonitor:
    """Capture the actual peak GPU pair demand on each recorded physics step."""

    def __init__(self, physics_context) -> None:
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        import omni.usd
        from pxr import PhysicsSchemaTools, Sdf

        self._stats_type = PhysicsSceneStats
        self._interface = get_physx_statistics_interface()
        self._stage_id = omni.usd.get_context().get_stage_id()
        self._scene_path = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(
            physics_context.prim_path))
        self.configured = {
            "found": int(physics_context.get_gpu_found_lost_aggregate_pairs_capacity()),
            "total": int(physics_context.get_gpu_total_aggregate_pairs_capacity()),
        }
        self.peaks = {"found": 0, "total": 0}
        self.sample_count = self.read_failures = 0
        self.scene_stats = {}
        self.sample()

    def sample(self) -> None:
        stats = self._stats_type()
        if not self._interface.get_physx_scene_statistics(
            self._stage_id, self._scene_path, stats):
            self.read_failures += 1
            return
        self.sample_count += 1
        self.peaks["found"] = max(self.peaks["found"], int(
            stats.gpu_mem_found_lost_aggregate_pairs))
        self.peaks["total"] = max(self.peaks["total"], int(
            stats.gpu_mem_total_aggregate_pairs))
        shape_fields = (
            "nb_box_shapes", "nb_capsule_shapes", "nb_cone_shapes",
            "nb_convex_shapes", "nb_cylinder_shapes", "nb_plane_shapes",
            "nb_sphere_shapes", "nb_trimesh_shapes",
        )
        self.scene_stats = {
            "rigid_body_count": int(stats.nb_dynamic_rigids + stats.nb_kinematic_rigids
                                    + stats.nb_static_rigids),
            "articulation_count": int(stats.nb_articulations),
            "aggregate_count": int(stats.nb_aggregates),
            "physx_collision_shape_count": sum(int(getattr(stats, key))
                                                for key in shape_fields),
        }

    def summary(self) -> dict[str, object]:
        return {
            "configured_gpu_found_lost_aggregate_pairs_capacity": self.configured["found"],
            "configured_gpu_total_aggregate_pairs_capacity": self.configured["total"],
            "observed_gpu_found_lost_aggregate_pairs_peak": self.peaks["found"],
            "observed_gpu_total_aggregate_pairs_peak": self.peaks["total"],
            "physx_statistics_sample_count": self.sample_count,
            "physx_statistics_read_failures": self.read_failures,
            "scene_counts": self.scene_stats,
        }


def audit_physx_log(
    path: Path, *, cutoff_bytes: int | None = None,
    required_marker: str | None = None,
) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError:
        return {"scan_complete": False, "capacity_warning_count": None, "sha256": None}
    if cutoff_bytes is not None:
        if (isinstance(cutoff_bytes, bool) or not isinstance(cutoff_bytes, int)
                or not 0 < cutoff_bytes <= len(payload)):
            return {"scan_complete": False, "capacity_warning_count": None,
                    "sha256": None, "audit_byte_count": None}
        payload = payload[:cutoff_bytes]
    marker_seen = bool(
        required_marker is None or (
            isinstance(required_marker, str)
            and required_marker.encode("ascii") in payload
        )
    )
    lines = payload.decode("utf-8", errors="replace").splitlines()
    capacity_warnings, physx_warnings, physx_errors = [], [], []
    requested = {"found": 0, "total": 0}
    for line in lines:
        lower = line.lower()
        is_physx_warning = "omni.physx" in lower and any(
            level in lower for level in ("[warning]", "[error]"))
        if is_physx_warning:
            physx_warnings.append(line)
        if "omni.physx" in lower and "[error]" in lower:
            physx_errors.append(line)
        match = _CAPACITY_REQUEST.search(line)
        capacity_issue = "simulation will miss interactions" in lower or bool(
            is_physx_warning and "capacity" in lower and any(word in lower for word in
                ("increase", "exceed", "insufficient", "overflow")))
        if capacity_issue:
            capacity_warnings.append(line)
        if match:
            key = "found" if match.group(1).lower().startswith("found") else "total"
            requested[key] = max(requested[key], int(match.group(2)))
    return {
        "scan_complete": marker_seen,
        "cutoff_marker_seen": marker_seen,
        "audit_byte_count": len(payload),
        "capacity_warning_count": len(capacity_warnings),
        "capacity_warning_lines": capacity_warnings,
        "all_physx_warning_lines": physx_warnings,
        "physx_error_lines": physx_errors,
        "requested_found_lost_peak": requested["found"],
        "requested_total_peak": requested["total"],
        "sha256": hashlib.sha256(payload).hexdigest(), "path": str(path),
    }


def synchronize_engine_log(path: Path) -> dict[str, object]:
    import carb

    marker = f"CARTS_V2_ENGINE_LOG_SYNC_{time.time_ns()}"
    carb.log_info(marker)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            payload = path.read_bytes()
            marker_start = payload.find(marker.encode("ascii"))
            if marker_start >= 0:
                line_end = payload.find(b"\n", marker_start)
                cutoff = line_end + 1 if line_end >= 0 else marker_start + len(marker)
                return {
                    "marker": marker, "marker_seen": True,
                    "audit_byte_count": cutoff,
                    "audit_boundary": "PROCESS_START_THROUGH_SYNC_MARKER",
                }
        except OSError:
            pass
        time.sleep(0.01)
    raise RuntimeError("exact Kit log did not synchronize before evidence audit")


def identity_hashes_match(trace: Mapping[str, object]) -> bool:
    binding, runtime = trace.get("evidence_binding", {}), trace.get("runtime", {})
    sources = runtime.get("source_sha256", {})
    return bool(
        binding.get("config_sha256") == trace.get("config_sha256")
        == runtime.get("config_sha256")
        and binding.get("registered_grasp_sha256")
        == runtime.get("registered_grasp_sha256")
        and binding.get("control_plan_sha256") == runtime.get("control_plan_sha256")
        and binding.get("scene_evidence_sha256")
        == runtime.get("scene_evidence_sha256")
        and binding.get("robot_asset_sha256") == runtime.get("robot_asset_sha256")
        and binding.get("object_asset_sha256") == runtime.get("object_asset_sha256")
        and binding.get("controller_source_sha256") == sources.get("controller.py")
        and binding.get("runner_source_sha256") == sources.get("run_grasp_lift.py")
        and binding.get("evaluator_source_sha256") == sources.get("evaluate_run.py")
        and binding.get("engine_health_source_sha256") == sources.get("engine_health.py")
        and binding.get("runtime_resources_sha256")
        == runtime.get("runtime_resources_sha256")
        and binding.get("capacity_audit_sha256")
        == runtime.get("capacity_audit_sha256")
    )


def finalize_engine_evaluation(evaluation, engine_runtime, log_path: Path):
    sync = engine_runtime.get("engine_log_sync", {})
    marker = sync.get("marker") if isinstance(sync, Mapping) else None
    cutoff = sync.get("audit_byte_count") if isinstance(sync, Mapping) else None
    log = audit_physx_log(log_path, cutoff_bytes=cutoff, required_marker=marker)
    marker_seen = bool(
        isinstance(marker, str) and marker.startswith("CARTS_V2_ENGINE_LOG_SYNC_")
        and sync.get("marker_seen") is True
        and sync.get("audit_boundary") == "PROCESS_START_THROUGH_SYNC_MARKER"
        and log.get("cutoff_marker_seen") is True)
    found_peak = max(int(engine_runtime[
        "observed_gpu_found_lost_aggregate_pairs_peak"]),
        int(log.get("requested_found_lost_peak", 0)))
    total_peak = max(int(engine_runtime[
        "observed_gpu_total_aggregate_pairs_peak"]),
        int(log.get("requested_total_peak", 0)))
    found_capacity = int(engine_runtime["configured_gpu_found_lost_aggregate_pairs_capacity"])
    total_capacity = int(engine_runtime["configured_gpu_total_aggregate_pairs_capacity"])
    engine_pass = bool(
        engine_runtime.get("gpu_backend_pass") is True
        and marker_seen
        and engine_runtime.get("physx_statistics_sample_count", 0) > 0
        and engine_runtime.get("physx_statistics_read_failures") == 0
        and log.get("scan_complete") is True
        and log.get("capacity_warning_count") == 0
        and not log.get("physx_error_lines")
        and found_peak < found_capacity and total_peak < total_capacity
    )
    controller = evaluation.get("controller_preflight_pass") is True
    identity = evaluation.get("identity_hash_check_pass") is True
    preflight_boundary = bool(
        evaluation.get("schema_version") in {
            "carts_grasp_v2_dynamic_evaluation_v2",
            "carts_grasp_v2_dynamic_evaluation_v3",
            "carts_grasp_v2_dynamic_evaluation_v4",
        }
        and evaluation.get("mode") == "preflight"
        and evaluation.get("hardware_authorized") is False
        and evaluation.get("formal_dynamic_pass") is False
        and evaluation.get("research_dynamic_pass") is False)
    accepted = bool(preflight_boundary and controller and engine_pass and identity)
    physical = bool(evaluation.get("nominal_diagnostic_pass"))
    nominal = bool(physical and evaluation.get("pad_surface_identity_verified") is True
                   and evaluation.get("accepted_preflight_bound") is True
                   and evaluation.get("truth_isolation_pass") is True
                   and engine_pass and identity)
    first_finger = bool(evaluation.get("mode") == "first-finger-diagnostic"
                        and evaluation.get("hardware_authorized") is False and evaluation.get("formal_dynamic_pass") is False
                        and evaluation.get("pad_surface_identity_verified") is True and evaluation.get("first_finger_contact_classification") == "ALLOWED_PAD_CONTACT"
                        and evaluation.get("controller_first_finger_diagnostic_pass") and evaluation.get("accepted_preflight_bound") is True
                        and evaluation.get("truth_isolation_pass") is True
                        and engine_pass and identity)
    evaluation.update({
        "preflight_pass": accepted, "engine_health_pass": engine_pass,
        "accepted_preflight_pass": accepted,
        "physx_capacity_warning_count": log.get("capacity_warning_count"),
        "configured_gpu_found_lost_aggregate_pairs_capacity": found_capacity,
        "configured_gpu_total_aggregate_pairs_capacity": total_capacity,
        "observed_gpu_found_lost_aggregate_pairs_peak": found_peak,
        "observed_gpu_total_aggregate_pairs_peak": total_peak,
        "engine_log_sha256": log.get("sha256"), "engine_log_path": log.get("path"),
        "engine_log_sync_marker": marker, "engine_log_marker_seen": marker_seen,
        "engine_log_audit_byte_count": log.get("audit_byte_count"),
        "engine_log_audit_boundary": "PROCESS_START_THROUGH_SYNC_MARKER",
        "physx_capacity_warning_lines": log.get("capacity_warning_lines", []),
        "all_physx_warning_lines": log.get("all_physx_warning_lines", []),
        "physx_error_lines": log.get("physx_error_lines", []),
        "controller_nominal_physical_pass": physical,
        "nominal_research_dynamic_pass": nominal,
        "first_finger_diagnostic_pass": first_finger,
        "research_dynamic_pass": bool(evaluation.get("research_dynamic_pass")
                                      and nominal),
        "engine_runtime": engine_runtime,
    })
    return evaluation


def pending_engine_fields(
    controller_preflight_pass: bool, identity_hash_check_pass: bool
) -> dict[str, object]:
    """Return the complete schema while engine evidence is still unavailable."""

    return {
        "controller_preflight_pass": bool(controller_preflight_pass),
        "engine_health_pass": False, "accepted_preflight_pass": False,
        "physx_capacity_warning_count": None,
        "configured_gpu_found_lost_aggregate_pairs_capacity": None,
        "configured_gpu_total_aggregate_pairs_capacity": None,
        "observed_gpu_found_lost_aggregate_pairs_peak": None,
        "observed_gpu_total_aggregate_pairs_peak": None,
        "engine_log_sha256": None, "engine_log_sync_marker": None,
        "engine_log_marker_seen": False, "engine_log_audit_byte_count": None,
        "engine_log_audit_boundary": None, "physx_error_lines": None,
        "identity_hash_check_pass": bool(identity_hash_check_pass),
    }


def preflight_is_accepted(document: Mapping[str, object]) -> bool:
    """Require the complete engine-backed preflight record; missing means reject."""

    if any(field not in document for field in ENGINE_EVIDENCE_FIELDS):
        return False
    binding = document.get("evidence_binding")
    required_binding = (
        "config_sha256", "registered_grasp_sha256", "control_plan_sha256",
        "runtime_resources_sha256", "capacity_audit_sha256",
        "scene_evidence_sha256", "object_asset_sha256", "robot_asset_sha256",
        "controller_source_sha256", "runner_source_sha256",
        "evaluator_source_sha256", "engine_health_source_sha256",
    )
    boundary = bool(
        document.get("schema_version") in {
            "carts_grasp_v2_dynamic_evaluation_v2",
            "carts_grasp_v2_dynamic_evaluation_v3",
            "carts_grasp_v2_dynamic_evaluation_v4",
        }
        and document.get("mode") == "preflight"
        and document.get("hardware_authorized") is False
        and document.get("formal_dynamic_pass") is False
        and document.get("research_dynamic_pass") is False
        and isinstance(binding, Mapping)
        and all(field in binding for field in required_binding)
        and all(_SHA256.fullmatch(str(binding[field])) for field in required_binding
                if field != "scene_evidence_sha256")
        and isinstance(binding.get("scene_evidence_sha256"), Mapping)
        and bool(binding.get("scene_evidence_sha256"))
        and all(_SHA256.fullmatch(str(value)) for value in
                binding.get("scene_evidence_sha256", {}).values()))
    controller = document.get("controller_preflight_pass") is True
    engine = document.get("engine_health_pass") is True
    identity = document.get("identity_hash_check_pass") is True
    accepted = document.get("accepted_preflight_pass") is True
    try:
        found_capacity = int(document[
            "configured_gpu_found_lost_aggregate_pairs_capacity"])
        total_capacity = int(document["configured_gpu_total_aggregate_pairs_capacity"])
        found_peak = int(document["observed_gpu_found_lost_aggregate_pairs_peak"])
        total_peak = int(document["observed_gpu_total_aggregate_pairs_peak"])
    except (TypeError, ValueError):
        return False
    log_sha = document.get("engine_log_sha256")
    marker = document.get("engine_log_sync_marker")
    audit_bytes = document.get("engine_log_audit_byte_count")
    synchronized_log = bool(
        isinstance(marker, str) and marker.startswith("CARTS_V2_ENGINE_LOG_SYNC_")
        and document.get("engine_log_marker_seen") is True
        and isinstance(audit_bytes, int) and not isinstance(audit_bytes, bool)
        and audit_bytes > 0
        and document.get("engine_log_audit_boundary")
        == "PROCESS_START_THROUGH_SYNC_MARKER"
        and document.get("physx_error_lines") == [])
    capacity_evidence = bool(document.get("physx_capacity_warning_count") == 0
        and 0 <= found_peak < found_capacity
        and 0 <= total_peak < total_capacity
        and isinstance(log_sha, str) and _SHA256.fullmatch(log_sha))
    expected = bool(boundary and controller and engine and identity
                    and synchronized_log and capacity_evidence)
    return accepted and expected


__all__ = [
    "ENGINE_EVIDENCE_FIELDS", "PhysxStatsMonitor", "audit_physx_log",
    "current_engine_log_path", "finalize_engine_evaluation", "gpu_backend_record",
    "gpu_world_parameters", "identity_hashes_match", "load_runtime_resources",
    "pending_engine_fields", "preflight_is_accepted", "synchronize_engine_log",
]
