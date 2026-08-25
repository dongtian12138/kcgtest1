"""Bind official GraspGenX files to V2 seeds without importing its runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    V2Inputs,
    allowed_face_domain_sha256,
)
from kcg_connector.grasp.robust.object_model import file_sha256

_SCHEMAS_BY_BACKEND = {
    "GRASPGENX": (
        "graspgenx_carts_proposals_v1",
        "kcg_graspgenx_descriptors_v1",
    ),
    "GRASPGENX_FULL_PALM": (
        "graspgenx_carts_full_palm_proposals_v2",
        "kcg_graspgenx_descriptors_v2",
    ),
}
_ROTATION_NUMERICAL_ATOL = 1e-6
_MERGED_KEEP_METHOD = "DESCRIPTOR_APPROACH_STRATA_THEN_6D_FARTHEST_FILL"
_CLOSURE_PHASE_RULE = "PER_PRESHAPE_LAST_SAMPLED_SELF_COLLISION_FREE_STATE"
@dataclass(frozen=True)
class AdaptedCandidate:
    """One legacy-compatible seed paired with its generator evidence."""

    seed: CandidateSeed
    evidence: Mapping[str, Any]

def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value

def _digest(value: Any, label: str, *, length: int = 64) -> str:
    parsed = str(value).lower()
    if len(parsed) != length or any(c not in "0123456789abcdef" for c in parsed):
        raise ValueError(f"{label} must be a {length}-character hexadecimal digest")
    return parsed
def _rigid(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be one finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_ROTATION_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=_ROTATION_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{label} rotation is not right handed")
    left, _singular, right = np.linalg.svd(rotation)
    canonical = left @ right
    if np.linalg.det(canonical) < 0.0:
        left[:, -1] *= -1.0
        canonical = left @ right
    result = np.array(matrix, copy=True)
    result[:3, :3] = canonical
    return result

def _descriptor_rows(
    inputs: V2Inputs, path: Path, expected_sha256: str, expected_schema: str,
) -> dict[str, dict[str, Any]]:
    if file_sha256(path) != _digest(expected_sha256, "descriptor manifest SHA-256"):
        raise ValueError("descriptor manifest SHA-256 mismatch")
    document = _json_object(path)
    schema = str(document.get("schema_version", ""))
    if schema != expected_schema or document.get("object_independent") is not True:
        raise ValueError("descriptor manifest identity changed")
    full_palm = schema == "kcg_graspgenx_descriptors_v2"
    if full_palm:
        if (
            document.get("palm_configuration_grid_rule")
            != "URDF_CLOSED_LIMITS_91_UNIFORM"
            or int(document.get("palm_configuration_count", 0)) != 91
            or document.get("conditioning_close_selection_role")
            != "GRASPGENX_CONDITIONING_ONLY_NOT_SEQUENTIAL_PHYSICAL_LIMIT"
        ):
            raise ValueError("full-palm descriptor grid/conditioning identity changed")
    else:
        configured_rule = inputs.config.section("candidate_generation").get(
            "maximum_closure_phase_rule"
        )
        if configured_rule != _CLOSURE_PHASE_RULE or document.get(
            "maximum_closure_phase_rule"
        ) != _CLOSURE_PHASE_RULE:
            raise ValueError("descriptor closure phase rule differs from frozen route")
    roster = inputs.repository_root / inputs.config.section("inputs")["collision_roster"]
    if file_sha256(inputs.hand_contract.contract_path) != document.get("hand_contract_sha256"):
        raise ValueError("descriptor hand contract does not match current hand")
    if file_sha256(roster) != document.get("collision_roster_sha256"):
        raise ValueError("descriptor collision roster does not match current hand")
    result: dict[str, dict[str, Any]] = {}
    for row in document.get("descriptors", ()):
        descriptor_id = str(row.get("descriptor_id", ""))
        if not descriptor_id or descriptor_id in result:
            raise ValueError("descriptor IDs must be unique and non-empty")
        hand_from_generator = _rigid(row["handbase_from_graspgenx_row_major"], "handbase_from_graspgenx")
        generator_from_hand = _rigid(row["graspgenx_from_handbase_row_major"], "graspgenx_from_handbase")
        if not np.allclose(hand_from_generator @ generator_from_hand, np.eye(4), atol=1e-9):
            raise ValueError("descriptor frame transforms are not inverse")
        open_map = {str(k): float(v) for k, v in row["open_joint_positions_rad"].items()}
        close_key = (
            "conditioning_close_joint_positions_rad"
            if full_palm
            else "close_joint_positions_rad"
        )
        close_map = {str(k): float(v) for k, v in row[close_key].items()}
        inputs.hand_model.resolve_joint_positions(open_map, enforce_limits=True)
        inputs.hand_model.resolve_joint_positions(close_map, enforce_limits=True)
        palm_configuration = float(
            row["palm_configuration_rad"]
            if full_palm
            else row.get("preshape_f1j1_rad", open_map["f1j1"])
        )
        if (
            not math.isfinite(palm_configuration)
            or not np.isclose(open_map["f1j1"], palm_configuration, atol=1.0e-12)
            or not np.isclose(open_map["f3j1"], palm_configuration, atol=1.0e-12)
        ):
            raise ValueError("descriptor palm configuration was lost or changed")
        fingertip = np.asarray(row["graspgenx_config"]["fingertip"], dtype=np.float64)
        if fingertip.shape != (3,) or not np.all(np.isfinite(fingertip)):
            raise ValueError("descriptor fingertip must be one finite 3-vector")
        sweep = row["graspgenx_config"]["sweep_volume"]
        open_extents = np.asarray(sweep["extents"], dtype=np.float64)
        open_offset = np.asarray(sweep["offset"], dtype=np.float64)
        half_extents = np.asarray(sweep["extents2"], dtype=np.float64)
        half_offset = np.asarray(sweep["offset2"], dtype=np.float64)
        if (
            any(value.shape != (3,) for value in (
                open_extents, open_offset, half_extents, half_offset
            ))
            or any(not np.all(np.isfinite(value)) for value in (
                open_extents, open_offset, half_extents, half_offset
            ))
            or np.any(open_extents <= 0.0)
            or np.any(half_extents <= 0.0)
        ):
            raise ValueError("descriptor sweep boxes must be positive and finite")
        result[descriptor_id] = {
            "row": row, "generator_from_hand": generator_from_hand,
            "fingertip": fingertip,
            "open_sweep_lower": open_offset - 0.5 * open_extents,
            "open_sweep_upper": open_offset + 0.5 * open_extents,
            "half_sweep_lower": half_offset - 0.5 * half_extents,
            "half_sweep_upper": half_offset + 0.5 * half_extents,
            "open_independent": tuple(open_map[name] for name in inputs.hand_model.independent_joint_names),
            "palm_configuration_rad": palm_configuration,
            "conditioning_close_phase": float(
                row["conditioning_close_phase"]
                if full_palm
                else row["maximum_closure_phase"]
            ),
            "physical_maximum_closure_phase": (
                None if full_palm else float(row["maximum_closure_phase"])
            ),
        }
    if not result:
        raise ValueError("descriptor manifest has no descriptor")
    if full_palm:
        limit = inputs.hand_model.independent_joint_limits["f1j1"]
        expected = np.linspace(limit.lower, limit.upper, 91)
        observed = np.asarray([
            result[key]["palm_configuration_rad"] for key in sorted(result)
        ])
        if len(result) != 91 or not np.allclose(observed, expected, atol=1.0e-12):
            raise ValueError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
    return result

def _validate_mesh(inputs: V2Inputs, payload: Mapping[str, Any], path: Path) -> str:
    actual = file_sha256(path)
    if actual != _digest(payload.get("standardized_mesh_sha256"), "mesh SHA-256"):
        raise ValueError("standardized mesh SHA-256 mismatch")
    if Path(str(payload.get("standardized_mesh"))).resolve() != path:
        raise ValueError("proposal references a different standardized mesh path")
    if payload.get("source_mesh_sha256") != inputs.object_contract.model.provenance.source_sha256:
        raise ValueError("proposal source mesh does not match the registered object")
    inference_from_object = _rigid(
        payload.get("inference_from_object_row_major"), "inference_from_object"
    )
    if (
        payload.get("inference_frame") != "FROZEN_SCENE_WORLD"
        or not np.allclose(
            inference_from_object,
            inputs.frozen_world_from_object,
            atol=1.0e-9,
            rtol=0.0,
        )
    ):
        raise ValueError("proposal inference frame differs from the frozen scene")
    with np.load(path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        allowed = np.asarray(archive["allowed_face_indices"], dtype=np.int64)
    registered = inputs.object_contract.model.mesh
    settings = inputs.config.section("candidate_generation")["graspgenx"]
    downstream_scope = payload.get(
        "downstream_collision_geometry_scope",
        payload.get("collision_geometry_scope"),
    )
    if not np.array_equal(faces, registered.faces) or not np.array_equal(vertices, registered.vertices_m):
        raise ValueError("standardized mesh geometry differs from the registered object")
    domain_sha = allowed_face_domain_sha256(len(faces), allowed)
    if (
        not np.array_equal(allowed, inputs.face_roles.allowed_face_indices)
        or payload.get("allowed_face_count") != len(allowed)
        or payload.get("allowed_face_domain_sha256") != domain_sha
        or payload.get("face_role_method") != inputs.face_roles.method
        or payload.get("proposal_conditioning_mode")
        != settings["proposal_conditioning_mode"]
        or downstream_scope != settings["downstream_collision_geometry_scope"]
    ):
        raise ValueError("proposal allowed-face identity differs from current V2 inputs")
    return actual

def _approach_stratum(row: tuple[Any, ...], inference_from_object: np.ndarray) -> str:
    if str(row[3].get("branch", "")) != "obb":
        return "diff"
    approach = (inference_from_object @ row[5])[:3, 2]
    if approach[2] < -0.5:
        return "obb_top"
    azimuth = float(np.mod(np.arctan2(approach[1], approach[0]), 2.0 * np.pi))
    return f"obb_side_{min(int(azimuth / (0.5 * np.pi)), 3)}"

def _select_six_d_diverse(
    rows: list[tuple[Any, ...]], position_scale_m: float,
    orientation_scale_rad: float, limit: int, inference_from_object: np.ndarray,
) -> list[tuple[Any, ...]]:
    """Preserve fixed descriptor/direction strata, then farthest-fill in 6-D."""

    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(rows) <= limit:
        return rows
    poses = np.asarray([item[6] for item in rows])
    translation = np.linalg.norm(
        poses[:, None, :3, 3] - poses[None, :, :3, 3], axis=2
    ) / position_scale_m
    rotation_trace = np.einsum(
        "aij,bij->ab", poses[:, :3, :3], poses[:, :3, :3]
    )
    rotation = np.arccos(np.clip((rotation_trace - 1.0) * 0.5, -1.0, 1.0))
    distance = np.maximum(translation, rotation / orientation_scale_rad)
    cells: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(rows):
        key = (item[1], _approach_stratum(item, inference_from_object))
        cells.setdefault(key, []).append(index)
    quota = limit // len(cells)
    selected = [index for key in sorted(cells) for index in cells[key][:quota]]
    if not selected:
        selected = [0]
    nearest = np.min(distance[:, selected], axis=1)
    while len(selected) < limit:
        nearest[selected] = -np.inf
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        nearest = np.minimum(nearest, distance[:, chosen])
    return [rows[index] for index in selected]

def _convert_and_deduplicate(
    payload: Mapping[str, Any], descriptors: Mapping[str, Mapping[str, Any]],
    translation_tolerance_m: float, rotation_tolerance_rad: float, maximum_candidates: int,
    *, require_palm_configuration: bool,
) -> list[tuple[Any, ...]]:
    converted = []
    seen_sources: set[tuple[str, int]] = set()
    for row in payload.get("proposals", ()):
        descriptor_id = str(row.get("descriptor_id", ""))
        raw_index = int(row.get("raw_index", -1))
        key = (descriptor_id, raw_index)
        score = float(row.get("score", math.nan))
        if (
            descriptor_id not in descriptors
            or raw_index < 0
            or key in seen_sources
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError("proposal source identity or score is invalid")
        seen_sources.add(key)
        descriptor = descriptors[descriptor_id]
        proposal_palm = row.get("palm_configuration_rad")
        if require_palm_configuration and proposal_palm is None:
            raise ValueError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
        if proposal_palm is not None and (
            not math.isfinite(float(proposal_palm))
            or not np.isclose(
                float(proposal_palm), descriptor["palm_configuration_rad"],
                atol=1.0e-12, rtol=0.0,
            )
        ):
            raise ValueError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
        object_from_generator = _rigid(row["object_from_graspgenx_row_major"], "object_from_graspgenx")
        object_from_hand = _rigid(
            object_from_generator @ descriptor["generator_from_hand"],
            "object_from_handbase",
        )
        fingertip = object_from_generator[:3, :3] @ descriptor["fingertip"] + object_from_generator[:3, 3]
        converted.append((score, descriptor_id, raw_index, row, descriptor, object_from_generator, object_from_hand, fingertip))
    converted.sort(key=lambda item: (-item[0], item[1], item[2]))

    by_descriptor: dict[str, list[tuple[Any, ...]]] = {}
    for item in converted:
        by_descriptor.setdefault(item[1], []).append(item)
    accepted_sources: set[tuple[str, int]] = set()
    for rows in by_descriptor.values():
        accepted_poses: list[np.ndarray] = []
        for item in rows:
            pose = item[6]
            duplicate = False
            if accepted_poses:
                previous = np.asarray(accepted_poses)
                translation = np.linalg.norm(previous[:, :3, 3] - pose[:3, 3], axis=1)
                traces = np.einsum(
                    "nij,ij->n", previous[:, :3, :3], pose[:3, :3]
                )
                rotation = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
                duplicate = bool(np.any((translation <= translation_tolerance_m)
                                        & (rotation <= rotation_tolerance_rad)))
            if not duplicate:
                accepted_poses.append(pose)
                accepted_sources.add((item[1], item[2]))
    unique = [item for item in converted if (item[1], item[2]) in accepted_sources]
    inference_from_object = _rigid(
        payload["inference_from_object_row_major"], "inference_from_object"
    )
    return _select_six_d_diverse(
        unique, translation_tolerance_m, rotation_tolerance_rad,
        int(maximum_candidates), inference_from_object,
    )

def _validate_inference_parameters(inputs: V2Inputs, payload: Mapping[str, Any]) -> None:
    settings = inputs.config.section("candidate_generation")["graspgenx"]
    actual = dict(payload.get("inference_parameters", {}))
    legacy_downstream_scope = actual.pop("collision_geometry_scope", None)
    if legacy_downstream_scope not in (
        None, settings["downstream_collision_geometry_scope"]
    ):
        raise ValueError("legacy downstream collision scope changed")
    expected = {
        "object_sample_point_count": int(settings["object_sample_point_count"]),
        "object_surface_sample_method": str(
            settings["object_surface_sample_method"]
        ),
        "proposal_conditioning_mode": str(settings["proposal_conditioning_mode"]),
        "num_grasps": int(settings["raw_target_per_descriptor_object"]),
        "keep_per_descriptor": int(settings["keep_per_descriptor_object"]),
        "proposal_keep_method": str(settings["proposal_keep_method"]),
        "proposal_visibility_method": str(settings["proposal_visibility_method"]),
        "grasp_threshold": float(settings["grasp_threshold"]),
        "topk_num_grasps": int(settings["topk_num_grasps"]),
        "moe_num_yaws": int(settings["moe_num_yaws"]),
        "moe_z_offsets_cm": list(settings["moe_z_offsets_cm"]),
        "moe_outlier_threshold": float(settings["moe_outlier_threshold"]),
        "moe_outlier_k": int(settings["moe_outlier_k"]),
        "moe_obb_mode": str(settings["moe_obb_mode"]),
        "moe_skip_obb_rule": str(settings["moe_skip_obb_rule"]),
        "moe_obb_density": str(settings["moe_obb_density"]),
        "moe_obb_position_spacing_cm": float(
            settings["moe_obb_position_spacing_cm"]
        ),
    }
    if actual != expected:
        raise ValueError("proposal inference parameters differ from frozen route")

def _adapt_rows(
    inputs: V2Inputs, accepted: list[tuple[Any, ...]], payload: Mapping[str, Any],
    *, commit: str, checkpoint: str, descriptor_sha: str, mesh_sha: str,
    proposal_sha: str, random_seed: int,
) -> tuple[AdaptedCandidate, ...]:
    face_tree = cKDTree(inputs.object_contract.model.mesh.face_centroids_m)
    result = []
    for index, item in enumerate(accepted):
        score, descriptor_id, raw_index, row, descriptor, raw_pose, pose, fingertip = item
        _distance, face_index = face_tree.query(fingertip)
        anchor = inputs.object_contract.model.mesh.face_centroids_m[int(face_index)]
        candidate_id = f"graspgenx_{index:03d}"
        seed = CandidateSeed(
            candidate_id=candidate_id,
            object_id=inputs.object_contract.object_id,
            anchor_face_index=int(face_index),
            anchor_position_object_m=tuple(float(v) for v in anchor),
            object_from_hand=tuple(float(v) for v in pose.ravel()),
            pregrasp_joint_positions_rad=descriptor["open_independent"],
            pregrasp_closure_phases=(0.0, 0.0, 0.0),
            source_sample_index=raw_index,
            generator_score=score,
            descriptor_id=descriptor_id,
            approach_direction_object=tuple(
                float(value) for value in raw_pose[:3, 2]
            ),
            maximum_closure_phase=descriptor["physical_maximum_closure_phase"],
            palm_configuration_rad=descriptor["palm_configuration_rad"],
        )
        descriptor_row = descriptor["row"]
        evidence = {
            "candidate_id": candidate_id,
            "object_id": inputs.object_contract.object_id,
            "descriptor_id": descriptor_id,
            "graspgenx_score": score,
            "graspgenx_branch": str(row.get("branch", "")),
            "raw_index": raw_index,
            "object_from_graspgenx_row_major": list(
                row["object_from_graspgenx_row_major"]
            ),
            "object_from_handbase_row_major": pose.ravel().tolist(),
            "transform_chain": "object_from_graspgenx @ graspgenx_from_handbase",
            "generator_inference_frame": payload["inference_frame"],
            "inference_from_object_row_major": list(
                payload["inference_from_object_row_major"]
            ),
            "graspgenx_from_handbase_row_major": descriptor["generator_from_hand"].ravel().tolist(),
            "open_joint_positions_rad": descriptor_row["open_joint_positions_rad"],
            "conditioning_close_joint_positions_rad": descriptor_row.get(
                "conditioning_close_joint_positions_rad",
                descriptor_row.get("close_joint_positions_rad"),
            ),
            "conditioning_close_phase": descriptor["conditioning_close_phase"],
            "close_joint_positions_rad": descriptor_row.get(
                "close_joint_positions_rad",
                descriptor_row.get("conditioning_close_joint_positions_rad"),
            ),
            "maximum_closure_phase": descriptor[
                "physical_maximum_closure_phase"
            ],
            "palm_configuration_rad": descriptor["palm_configuration_rad"],
            "sequential_physical_maximum_closure_phase": (
                descriptor["physical_maximum_closure_phase"]
            ),
            "generator_commit": commit,
            "checkpoint_sha256": checkpoint,
            "source_mesh_sha256": payload["source_mesh_sha256"],
            "standardized_mesh_sha256": mesh_sha,
            "object_point_cloud_sha256": _digest(
                payload["object_point_cloud_sha256"], "object point-cloud SHA-256"
            ),
            "allowed_face_domain_sha256": payload["allowed_face_domain_sha256"],
            "generator_geometry_scope": payload["proposal_conditioning_mode"],
            "downstream_collision_geometry_scope": payload.get(
                "downstream_collision_geometry_scope",
                payload.get("collision_geometry_scope"),
            ),
            "downstream_collision_claim_scope": (
                "V2_CONTROL_STATE_FCL_NOT_OFFICIAL_GRASPGENX_COLLISION_FILTER"
            ),
            "descriptor_manifest_sha256": descriptor_sha,
            "proposal_file_sha256": proposal_sha,
            "random_seed": int(random_seed),
            "open_sweep_claim_scope": "POINT_CLOUD_VISIBILITY_DIAGNOSTIC_NOT_SELECTION_OR_COLLISION_PROOF",
            "merged_keep_method": _MERGED_KEEP_METHOD,
            "merged_keep_stratum": _approach_stratum(
                item, _rigid(
                    payload["inference_from_object_row_major"],
                    "inference_from_object",
                ),
            ),
        }
        result.append(AdaptedCandidate(seed, evidence))
    return tuple(result)

def load_graspgenx_candidates(
    inputs: V2Inputs, proposal_path: Path | str, descriptor_manifest_path: Path | str,
    standardized_mesh_path: Path | str, *, expected_generator_commit: str,
    expected_checkpoint_sha256: str, expected_random_seed: int,
    expected_descriptor_manifest_sha256: str,
    translation_tolerance_m: float, rotation_tolerance_rad: float,
    maximum_candidates: int = 256,
) -> tuple[AdaptedCandidate, ...]:
    """Validate, transform and 6-D de-duplicate official file proposals."""

    tolerances = (translation_tolerance_m, rotation_tolerance_rad)
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("6-D de-duplication tolerances must be positive")
    if not 1 <= int(maximum_candidates) <= 5824:
        raise ValueError("maximum_candidates must lie in [1, 5824]")
    proposals_path = Path(proposal_path).resolve()
    descriptor_path = Path(descriptor_manifest_path).resolve()
    mesh_path = Path(standardized_mesh_path).resolve()
    payload = _json_object(proposals_path)
    commit = _digest(expected_generator_commit, "generator commit", length=40)
    checkpoint = _digest(expected_checkpoint_sha256, "checkpoint SHA-256")
    backend = str(inputs.config.section("candidate_generation").get("backend", ""))
    schemas = _SCHEMAS_BY_BACKEND.get(backend)
    if schemas is None:
        raise ValueError("unsupported GraspGenX candidate backend")
    proposal_schema, descriptor_schema = schemas
    if payload.get("schema_version") != proposal_schema or payload.get("object_id") != inputs.object_contract.object_id:
        raise ValueError("proposal schema/object identity changed")
    if payload.get("generator_commit") != commit or payload.get("checkpoint_sha256") != checkpoint:
        raise ValueError("proposal generator/checkpoint identity mismatch")
    if payload.get("random_seed") != int(expected_random_seed):
        raise ValueError("proposal random seed mismatch")
    if payload.get("planner") != "graspmoe" or payload.get("model_loaded_once") is not True:
        raise ValueError("proposal inference method identity changed")
    if payload.get("model_load_count") != 1:
        raise ValueError("official model was not loaded exactly once")
    _digest(payload.get("object_point_cloud_sha256"), "object point-cloud SHA-256")
    merged_method = inputs.config.section("candidate_generation")["graspgenx"].get(
        "merged_keep_method"
    )
    if merged_method != _MERGED_KEEP_METHOD:
        raise ValueError("merged proposal keep method differs from the frozen route")
    _validate_inference_parameters(inputs, payload)
    descriptor_sha = _digest(payload.get("descriptor_manifest_sha256"), "descriptor SHA-256")
    if descriptor_sha != _digest(
        expected_descriptor_manifest_sha256, "expected descriptor SHA-256"
    ):
        raise ValueError("proposal descriptor identity differs from frozen route")
    descriptors = _descriptor_rows(
        inputs, descriptor_path, descriptor_sha, descriptor_schema
    )
    mesh_sha = _validate_mesh(inputs, payload, mesh_path)
    accepted = _convert_and_deduplicate(
        payload, descriptors, translation_tolerance_m,
        rotation_tolerance_rad, int(maximum_candidates),
        require_palm_configuration=backend == "GRASPGENX_FULL_PALM",
    )
    return _adapt_rows(
        inputs, accepted, payload, commit=commit, checkpoint=checkpoint,
        descriptor_sha=descriptor_sha, mesh_sha=mesh_sha,
        proposal_sha=file_sha256(proposals_path), random_seed=expected_random_seed,
    )


__all__ = ["AdaptedCandidate", "load_graspgenx_candidates"]
