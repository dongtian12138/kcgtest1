"""Compose a truth-free yaw-free TE transport pose with the frozen grasp.

The visual provider supplies translation and an oriented connector axis.  The
coupling-nut grasp is axisymmetric, so the missing key yaw is replaced by one
versioned geometric gauge: project frozen world +X onto the axis-normal plane
(world +Y is the declared degenerate fallback).  This is not an assembly-key
estimate and cannot authorize receptacle contact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


RELATION_SCHEMA_VERSION = "kcg_te_transport_grasp_relation_v1"
TARGET_SCHEMA_VERSION = "kcg_te_visual_transport_target_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_file(repository: Path, value: str, label: str) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (repository / raw).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def load_transport_grasp_relation(
    relation_path: Path | str, repository_root: Path | str
) -> tuple[Mapping[str, Any], Path]:
    repository = Path(repository_root).expanduser().resolve()
    path = _repository_file(repository, str(relation_path), "relation")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema_version") != (
        RELATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported transport grasp relation")
    if document.get("reference_part") != "CouplingNut":
        raise ValueError("transport relation must reference CouplingNut")
    if document.get("hardware_authorized") is not False:
        raise ValueError("transport relation must remain simulation-only")
    evidence = document["source_evidence"]
    for path_name, sha_name in (
        ("selected_config", "selected_config_sha256"),
        ("nominal_trace", "nominal_trace_sha256"),
    ):
        evidence_path = _repository_file(
            repository, evidence[path_name], f"relation {path_name}"
        )
        if _sha256(evidence_path) != evidence[sha_name]:
            raise ValueError(f"relation {path_name} identity differs")
    scope = document["scope"]
    if scope.get("transport_grasp_planning_allowed") is not True or any(
        scope.get(name) is not False
        for name in (
            "assembly_key_pose_consumption_allowed",
            "receptacle_contact_allowed",
            "insertion_allowed",
            "locking_allowed",
            "online_object_truth_allowed",
            "semantic_or_instance_truth_allowed",
        )
    ):
        raise ValueError("transport relation scope is not fail-closed")
    return document, path


def _yaw_free_world_from_object(
    position_xyz_m: Any,
    outward_axis_world: Any,
    gauge: Mapping[str, Any],
) -> np.ndarray:
    position = np.asarray(position_xyz_m, dtype=np.float64)
    axis = np.asarray(outward_axis_world, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("transport position must be one finite vector3")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("transport axis must be one finite vector3")
    norm = float(np.linalg.norm(axis))
    if not math.isclose(norm, 1.0, abs_tol=1.0e-6, rel_tol=0.0):
        raise ValueError("transport axis must be unit length")
    axis /= norm
    reference = np.asarray(gauge["primary_world_reference"], dtype=np.float64)
    projected = reference - float(reference @ axis) * axis
    if float(np.linalg.norm(projected)) < 1.0e-6:
        reference = np.asarray(
            gauge["degenerate_fallback_world_reference"], dtype=np.float64
        )
        projected = reference - float(reference @ axis) * axis
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm < 1.0e-6:
        raise ValueError("yaw-free gauge is degenerate for the observed axis")
    x_axis = projected / projected_norm
    y_axis = np.cross(axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, axis))
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-9
    ) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-9):
        raise RuntimeError("yaw-free object frame is not right-handed")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def build_visual_transport_target(
    *,
    provider_result_path: Path | str,
    relation_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    repository = Path(repository_root).expanduser().resolve()
    relation, resolved_relation_path = load_transport_grasp_relation(
        relation_path, repository
    )
    provider_path = _repository_file(
        repository, str(provider_result_path), "provider result"
    )
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    if provider.get("truth_flags", {}).get("uses_object_pose_truth") is not False:
        raise ValueError("provider result does not disclose truth isolation")
    if any(
        provider.get("truth_flags", {}).get(name) is not False
        for name in (
            "uses_semantic_truth",
            "uses_instance_truth",
            "uses_prim_transform",
            "uses_contact_truth",
        )
    ):
        raise ValueError("provider result used a forbidden online channel")
    transport = provider.get("transport_grasp_pose")
    if not isinstance(transport, Mapping):
        raise ValueError("provider result has no transport_grasp_pose")
    if (
        transport.get("status") != "OBSERVED_AXIS_POSITION_YAW_FREE"
        or transport.get("target_part") != "CouplingNut"
        or transport.get("main_key_required") is not False
        or transport.get("yaw_status")
        != "UNOBSERVED_FREE_FOR_AXISYMMETRIC_NUT_TRANSPORT_ONLY"
        or transport.get("transport_grasp_planning_input_available") is not True
        or transport.get("receptacle_contact_allowed") is not False
    ):
        raise ValueError("provider transport pose is not eligible for yaw-free planning")
    world_from_object = _yaw_free_world_from_object(
        transport["position_xyz_m"],
        transport["outward_axis_world"],
        relation["yaw_free_object_frame_gauge"],
    )
    object_from_hand = np.asarray(
        relation["transform"]["object_from_hand_base_row_major"],
        dtype=np.float64,
    ).reshape(4, 4)
    world_from_hand = world_from_object @ object_from_hand
    approach_object = np.asarray(
        relation["transform"]["approach_direction_object"], dtype=np.float64
    )
    approach_world = world_from_object[:3, :3] @ approach_object
    result = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "object_id": relation["object_id"],
        "capture_id": provider["capture_id"],
        "provider_result": {
            "path": str(provider_path.relative_to(repository)),
            "sha256": _sha256(provider_path),
            "transport_field": "transport_grasp_pose",
            "assembly_key_pose_consumed": False,
        },
        "transport_grasp_relation": {
            "path": str(resolved_relation_path.relative_to(repository)),
            "sha256": _sha256(resolved_relation_path),
            "relation_id": relation["relation_id"],
        },
        "yaw_free_gauge": dict(relation["yaw_free_object_frame_gauge"]),
        "world_from_transport_object_row_major": world_from_object.ravel().tolist(),
        "object_from_hand_base_row_major": object_from_hand.ravel().tolist(),
        "world_from_hand_base_target_row_major": world_from_hand.ravel().tolist(),
        "approach_direction_world": approach_world.tolist(),
        "robot_consumer_contract": {
            "runner_argument": "--visual-transport-target",
            "motion_plan_field": "world_from_hand_base_target",
            "matrix_composition": "world_T_transport_object @ object_T_hand_base",
            "target_must_equal_composed_matrix": True,
        },
        "authorization": {
            "simulation_only": True,
            "hardware_authorized": False,
            "transport_pregrasp_planning_input": True,
            "robot_motion_authorized_by_this_file": False,
            "receptacle_contact_authorized": False,
            "insertion_authorized": False,
            "locking_authorized": False,
        },
        "truth_flags": {
            "uses_object_pose_truth": False,
            "uses_semantic_truth": False,
            "uses_instance_truth": False,
            "uses_assembly_key_pose": False,
        },
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def load_visual_transport_target(
    target_path: Path | str, repository_root: Path | str
) -> tuple[dict[str, Any], Path]:
    repository = Path(repository_root).expanduser().resolve()
    path = _repository_file(repository, str(target_path), "visual transport target")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise ValueError("unsupported visual transport target")
    authorization = document.get("authorization", {})
    if (
        authorization.get("transport_pregrasp_planning_input") is not True
        or authorization.get("robot_motion_authorized_by_this_file") is not False
        or authorization.get("receptacle_contact_authorized") is not False
    ):
        raise ValueError("visual transport target authorization is invalid")
    provider = _repository_file(
        repository, document["provider_result"]["path"], "bound provider result"
    )
    relation = _repository_file(
        repository,
        document["transport_grasp_relation"]["path"],
        "bound transport relation",
    )
    if _sha256(provider) != document["provider_result"]["sha256"]:
        raise ValueError("bound provider result changed")
    if _sha256(relation) != document["transport_grasp_relation"]["sha256"]:
        raise ValueError("bound transport relation changed")
    rebuilt = build_visual_transport_target(
        provider_result_path=provider,
        relation_path=relation,
        repository_root=repository,
    )
    if document != rebuilt:
        raise ValueError(
            "visual transport target is not the deterministic composition of "
            "its bound provider result and grasp relation"
        )
    return document, path


def compare_visual_transport_targets(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    if first.get("schema_version") != TARGET_SCHEMA_VERSION or second.get(
        "schema_version"
    ) != TARGET_SCHEMA_VERSION:
        raise ValueError("comparison inputs must be visual transport targets")
    if first["transport_grasp_relation"]["sha256"] != second[
        "transport_grasp_relation"
    ]["sha256"]:
        raise ValueError("comparison targets use different grasp relations")
    first_object = np.asarray(
        first["world_from_transport_object_row_major"], dtype=np.float64
    ).reshape(4, 4)
    second_object = np.asarray(
        second["world_from_transport_object_row_major"], dtype=np.float64
    ).reshape(4, 4)
    first_hand = np.asarray(
        first["world_from_hand_base_target_row_major"], dtype=np.float64
    ).reshape(4, 4)
    second_hand = np.asarray(
        second["world_from_hand_base_target_row_major"], dtype=np.float64
    ).reshape(4, 4)
    relation = np.asarray(
        first["object_from_hand_base_row_major"], dtype=np.float64
    ).reshape(4, 4)
    object_difference = float(np.linalg.norm(first_object - second_object))
    target_difference = float(np.linalg.norm(first_hand - second_hand))
    invertible = abs(float(np.linalg.det(relation))) > 1.0e-12
    changed = bool(object_difference > 1.0e-12 and target_difference > 1.0e-12)
    return {
        "schema_version": "kcg_te_visual_transport_target_comparison_v1",
        "first_capture_id": first["capture_id"],
        "second_capture_id": second["capture_id"],
        "provider_results_differ": first["provider_result"]["sha256"]
        != second["provider_result"]["sha256"],
        "world_from_transport_object_frobenius_difference": object_difference,
        "world_from_hand_base_target_frobenius_difference": target_difference,
        "world_from_hand_base_translation_difference_m": (
            second_hand[:3, 3] - first_hand[:3, 3]
        ).tolist(),
        "object_from_hand_base_determinant": float(np.linalg.det(relation)),
        "right_multiplication_is_injective": invertible,
        "proof": (
            "for invertible O, W1@O == W2@O implies W1 == W2; "
            "therefore distinct visual W targets necessarily remain distinct"
        ),
        "planned_world_targets_differ": changed,
        "would_change_consumer_target_if_accepted": changed,
        "robot_command_consumed": False,
        "dynamic_motion_authorized": False,
        "assembly_key_pose_consumed": False,
        "receptacle_contact_authorized": False,
        "evidence_limit": (
            "OFFLINE_INJECTIVITY_ONLY; BOTH TARGETS REQUIRE AN INDEPENDENT "
            "SAFE PREFLIGHT BEFORE ROBOT MOTION"
        ),
    }


__all__ = [
    "RELATION_SCHEMA_VERSION",
    "TARGET_SCHEMA_VERSION",
    "build_visual_transport_target",
    "compare_visual_transport_targets",
    "load_transport_grasp_relation",
    "load_visual_transport_target",
]
