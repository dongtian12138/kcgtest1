from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.graspgenx_adapter import (
    load_graspgenx_candidates,
    summarize_six_d_coverage,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, ClosurePrediction, FastFilterResult, TaskQualityResult,
    write_standardized_object_manifest,
)
from kcg_connector.grasp.carts_v2.selector import select_candidate_rankings
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract


ROOT = Path(__file__).resolve().parents[4]
HAND_CONTRACT = ROOT / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
COLLISION_ROSTER = ROOT / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
OBJECT_ID = "current_d38999_26kj61sn_public_spec"
GENERATOR_COMMIT = "b" * 40
CHECKPOINT_SHA = "c" * 64
SEED = 20260824
INFERENCE_PARAMETERS = {
    "object_sample_point_count": 2048,
    "object_surface_sample_method": "TRIMESH_FULL_REGISTERED_MESH_SAMPLE_EXPLICIT_SEED",
    "proposal_conditioning_mode": "REGISTERED_FULL_OBJECT_MESH_SURFACE_POINT_CLOUD",
    "collision_geometry_scope": "FULL_REGISTERED_OBJECT_MESH",
    "num_grasps": 256,
    "keep_per_descriptor": 128,
    "proposal_keep_method": "HIGHEST_SCORE_PER_DESCRIPTOR",
    "proposal_visibility_method": "OFFICIAL_OPEN_OR_HALF_SWEEP_POINT_CLOUD_DIAGNOSTIC_ONLY",
    "grasp_threshold": -1.0,
    "topk_num_grasps": -1,
    "moe_num_yaws": 36,
    "moe_z_offsets_cm": [-2, 0],
    "moe_outlier_threshold": 0.014,
    "moe_outlier_k": 20,
    "moe_obb_mode": "advanced",
    "moe_skip_obb_rule": "auto",
    "moe_obb_density": "dense-topandside",
    "moe_obb_position_spacing_cm": 1.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Config:
    def section(self, name: str):
        if name == "inputs":
            return {"collision_roster": str(COLLISION_ROSTER.relative_to(ROOT))}
        if name == "candidate_generation":
            return {
                "backend": "GRASPGENX",
                "maximum_closure_phase_rule": (
                    "PER_PRESHAPE_LAST_SAMPLED_SELF_COLLISION_FREE_STATE"
                ),
                "deduplication": {
                    "palm_position_m": 0.001,
                    "palm_orientation_rad": 0.01,
                },
                "graspgenx": {
                    "object_sample_point_count": 2048,
                    "object_surface_sample_method": (
                        "TRIMESH_FULL_REGISTERED_MESH_SAMPLE_EXPLICIT_SEED"
                    ),
                "proposal_conditioning_mode": (
                    "REGISTERED_FULL_OBJECT_MESH_SURFACE_POINT_CLOUD"
                ),
                    "downstream_collision_geometry_scope": (
                        "FULL_REGISTERED_OBJECT_MESH"
                    ),
                    "raw_target_per_descriptor_object": 256,
                    "keep_per_descriptor_object": 128,
                    "proposal_keep_method": (
                        "HIGHEST_SCORE_PER_DESCRIPTOR"
                    ),
                    "merged_keep_method": (
                        "DESCRIPTOR_APPROACH_STRATA_THEN_6D_FARTHEST_FILL"
                    ),
                    **{
                        key: value
                        for key, value in INFERENCE_PARAMETERS.items()
                        if key not in {
                            "object_sample_point_count", "num_grasps",
                            "keep_per_descriptor", "collision_geometry_scope",
                        }
                    },
                },
            }
        raise AssertionError(name)


@pytest.fixture()
def adapter_case(tmp_path: Path):
    contract = load_carts_hand_contract(HAND_CONTRACT, repository_root=ROOT)
    hand = contract.build_hand_model()
    vertices = np.asarray(
        ((-0.02, -0.02, 0.0), (0.02, -0.02, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, 0.04)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int64)
    mesh = SimpleNamespace(
        vertices_m=vertices,
        faces=faces,
        face_centroids_m=np.mean(vertices[faces], axis=1),
    )
    model = SimpleNamespace(
        mesh=mesh,
        provenance=SimpleNamespace(source_sha256="a" * 64),
        assembly_axis_origin_m=np.zeros(3),
    )
    inputs = SimpleNamespace(
        repository_root=ROOT,
        config=_Config(),
        object_contract=SimpleNamespace(
            object_id=OBJECT_ID, model=model,
            task_frame_rotation_object=np.eye(3), characteristic_radius_m=0.02,
        ),
        hand_contract=contract,
        hand_model=hand,
        frozen_world_from_object=np.eye(4),
        face_roles=SimpleNamespace(
            allowed_face_indices=np.asarray((0, 1, 2), dtype=np.int64),
            allowed_area_m2=0.001,
            method="TASK_AXIS_OUTER_ENVELOPE_V1",
        ),
    )
    objects_path = tmp_path / "objects.json"
    objects = write_standardized_object_manifest(
        {OBJECT_ID: inputs}, tmp_path / "meshes", objects_path, 2048
    )
    object_row = objects["objects"][0]

    lower, upper = hand.joint_limit_vectors()
    names = tuple(hand.independent_joint_names)
    open_values = lower.copy()
    open_values[names.index("f1j1")] = 0.628
    close_values = open_values.copy()
    close_values[1:] = lower[1:] + 0.5 * (upper[1:] - lower[1:])
    open_map = {
        name: value
        for name, value in hand.resolve_joint_positions(open_values).items()
        if hand.joints[name].movable
    }
    close_map = {
        name: value
        for name, value in hand.resolve_joint_positions(close_values).items()
        if hand.joints[name].movable
    }
    angle = np.deg2rad(30.0)
    generator_from_hand = np.eye(4)
    generator_from_hand[:3, :3] = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    descriptor = {
        "descriptor_id": "kcg_3f_preshape_00",
        "maximum_closure_phase": 0.5,
        "open_joint_positions_rad": open_map,
        "close_joint_positions_rad": close_map,
        "handbase_from_graspgenx_row_major": np.linalg.inv(generator_from_hand).ravel().tolist(),
        "graspgenx_from_handbase_row_major": generator_from_hand.ravel().tolist(),
        "graspgenx_config": {
            "fingertip": [0.0, 0.0, 0.02],
            "sweep_volume": {
                "extents": [2.0, 2.0, 2.0],
                "offset": [0.0, 0.0, 0.0],
                "extents2": [2.0, 2.0, 2.0],
                "offset2": [0.0, 0.0, 0.0],
            },
        },
    }
    descriptor_document = {
        "schema_version": "kcg_graspgenx_descriptors_v1",
        "object_independent": True,
        "maximum_closure_phase_rule": (
            "PER_PRESHAPE_LAST_SAMPLED_SELF_COLLISION_FREE_STATE"
        ),
        "hand_contract_sha256": _sha256(HAND_CONTRACT),
        "collision_roster_sha256": _sha256(COLLISION_ROSTER),
        "descriptors": [descriptor],
    }
    descriptor_path = tmp_path / "descriptors.json"
    descriptor_path.write_text(json.dumps(descriptor_document), encoding="utf-8")

    raw_pose = np.eye(4)
    raw_pose[:3, 3] = (0.01, -0.02, 0.0)
    moved_pose = raw_pose.copy()
    moved_pose[0, 3] += 0.02
    payload = {
        "schema_version": "graspgenx_carts_proposals_v1",
        "object_id": OBJECT_ID,
        "generator_commit": GENERATOR_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_mesh_sha256": object_row["source_mesh_sha256"],
        "standardized_mesh": object_row["standardized_mesh_npz"],
        "standardized_mesh_sha256": object_row["standardized_mesh_sha256"],
        "object_point_cloud_sha256": "1" * 64,
        "full_object_point_cloud_sha256": "2" * 64,
        "proposal_conditioning_mode": "REGISTERED_FULL_OBJECT_MESH_SURFACE_POINT_CLOUD",
        "collision_geometry_scope": "FULL_REGISTERED_OBJECT_MESH",
        "allowed_face_count": object_row["allowed_face_count"],
        "allowed_surface_area_m2": object_row["allowed_surface_area_m2"],
        "allowed_face_domain_sha256": object_row["allowed_face_domain_sha256"],
        "face_role_method": object_row["face_role_method"],
        "inference_frame": object_row["inference_frame"],
        "inference_from_object_row_major": object_row[
            "inference_from_object_row_major"
        ],
        "descriptor_manifest_sha256": _sha256(descriptor_path),
        "random_seed": SEED,
        "planner": "graspmoe",
        "model_loaded_once": True,
        "model_load_count": 1,
        "inference_parameters": INFERENCE_PARAMETERS,
        "proposals": [
            {"raw_index": 0, "score": 0.8, "branch": "model", "descriptor_id": descriptor["descriptor_id"], "object_from_graspgenx_row_major": raw_pose.ravel().tolist()},
            {"raw_index": 1, "score": 0.9, "branch": "model", "descriptor_id": descriptor["descriptor_id"], "object_from_graspgenx_row_major": raw_pose.ravel().tolist()},
            {"raw_index": 2, "score": 0.7, "branch": "obb", "descriptor_id": descriptor["descriptor_id"], "object_from_graspgenx_row_major": moved_pose.ravel().tolist()},
        ],
    }
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    return SimpleNamespace(
        inputs=inputs,
        descriptor=descriptor,
        descriptor_path=descriptor_path,
        mesh_path=Path(object_row["standardized_mesh_npz"]),
        payload=payload,
        proposal_path=proposal_path,
        raw_pose=raw_pose,
        generator_from_hand=generator_from_hand,
        open_values=open_values,
    )


def _load(case, *, maximum_candidates=256, expected_descriptor_sha=None):
    return load_graspgenx_candidates(
        case.inputs,
        case.proposal_path,
        case.descriptor_path,
        case.mesh_path,
        expected_generator_commit=GENERATOR_COMMIT,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_random_seed=SEED,
        expected_descriptor_manifest_sha256=(
            expected_descriptor_sha or _sha256(case.descriptor_path)
        ),
        translation_tolerance_m=0.001,
        rotation_tolerance_rad=0.01,
        maximum_candidates=maximum_candidates,
    )


def test_standardized_mesh_transform_evidence_and_six_d_dedup(adapter_case) -> None:
    candidates = _load(adapter_case)
    assert len(candidates) == 2
    first = candidates[0]
    assert first.evidence["raw_index"] == 1
    assert np.allclose(
        first.seed.object_from_hand_matrix(),
        adapter_case.raw_pose @ adapter_case.generator_from_hand,
    )
    assert np.allclose(first.seed.pregrasp_joint_positions_rad, adapter_case.open_values)
    assert first.evidence["close_joint_positions_rad"] == adapter_case.descriptor["close_joint_positions_rad"]
    assert first.evidence["checkpoint_sha256"] == CHECKPOINT_SHA
    assert first.evidence["transform_chain"] == "object_from_graspgenx @ graspgenx_from_handbase"


def test_generator_checkpoint_object_and_seed_identity_fail_closed(adapter_case) -> None:
    for key, invalid in (
        ("generator_commit", "d" * 40),
        ("checkpoint_sha256", "e" * 64),
        ("object_point_cloud_sha256", "not-a-digest"),
        ("allowed_face_domain_sha256", "f" * 64),
        ("proposal_conditioning_mode", "FULL_OBJECT_POINT_CLOUD"),
        ("object_id", "different_object"),
        ("random_seed", SEED + 1),
    ):
        payload = copy.deepcopy(adapter_case.payload)
        payload[key] = invalid
        adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            _load(adapter_case)


def test_official_fp32_rotation_is_preserved_as_evidence_and_canonicalized(
    adapter_case,
) -> None:
    payload = copy.deepcopy(adapter_case.payload)
    observed = np.asarray(
        (
            (0.000471949577, 0.0240239743, -0.999711215, 0.300700488),
            (0.660896838, -0.750267267, -0.0177176557, -0.193913333),
            (-0.750476480, -0.660697639, -0.0162312984, 0.000155728076),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    payload["proposals"] = [copy.deepcopy(payload["proposals"][0])]
    payload["proposals"][0]["object_from_graspgenx_row_major"] = (
        observed.ravel().tolist()
    )
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")

    candidate = _load(adapter_case)[0]

    assert np.array_equal(
        np.asarray(candidate.evidence["object_from_graspgenx_row_major"]).reshape(4, 4),
        observed,
    )
    rotation = candidate.seed.object_from_hand_matrix()[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-12)


def test_descriptor_identity_and_fixed_inference_parameters_fail_closed(adapter_case) -> None:
    with pytest.raises(ValueError, match="frozen route"):
        _load(adapter_case, expected_descriptor_sha="f" * 64)
    payload = copy.deepcopy(adapter_case.payload)
    payload["inference_parameters"]["num_grasps"] = 255
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inference parameters"):
        _load(adapter_case)


def test_descriptor_closure_phase_rule_mismatch_fails_closed(adapter_case) -> None:
    document = json.loads(adapter_case.descriptor_path.read_text(encoding="utf-8"))
    document["maximum_closure_phase_rule"] = "PER_DESCRIPTOR_LEGACY_RULE"
    adapter_case.descriptor_path.write_text(json.dumps(document), encoding="utf-8")
    payload = copy.deepcopy(adapter_case.payload)
    payload["descriptor_manifest_sha256"] = _sha256(adapter_case.descriptor_path)
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="closure phase rule"):
        _load(adapter_case)


def test_merged_limit_preserves_descriptor_coverage_before_physics(adapter_case) -> None:
    document = json.loads(adapter_case.descriptor_path.read_text(encoding="utf-8"))
    second = copy.deepcopy(document["descriptors"][0])
    second["descriptor_id"] = "kcg_3f_preshape_01"
    document["descriptors"].append(second)
    adapter_case.descriptor_path.write_text(json.dumps(document), encoding="utf-8")
    payload = copy.deepcopy(adapter_case.payload)
    payload["descriptor_manifest_sha256"] = _sha256(adapter_case.descriptor_path)
    second_pose = np.array(adapter_case.raw_pose, copy=True)
    second_pose[1, 3] += 0.03
    payload["proposals"].append(
        {
            "raw_index": 0,
            "score": 0.01,
            "branch": "model",
            "descriptor_id": second["descriptor_id"],
            "object_from_graspgenx_row_major": second_pose.ravel().tolist(),
        }
    )
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    candidates = _load(adapter_case, maximum_candidates=2)
    assert {row.evidence["descriptor_id"] for row in candidates} == {
        "kcg_3f_preshape_00", "kcg_3f_preshape_01",
    }


def test_merged_limit_preserves_approach_strata_before_farthest_fill(
    adapter_case,
) -> None:
    payload = copy.deepcopy(adapter_case.payload)
    poses = []
    for raw_index, (branch, direction, score) in enumerate((
        ("model", (0.0, 0.0, 1.0), 0.9),
        ("obb", (0.0, 0.0, -1.0), 0.8),
        ("obb", (1.0, 0.0, 0.0), 0.7),
    )):
        pose = np.eye(4)
        if direction == (0.0, 0.0, -1.0):
            pose[:3, :3] = np.diag((1.0, -1.0, -1.0))
        elif direction == (1.0, 0.0, 0.0):
            pose[:3, :3] = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
        pose[:3, 3] = (0.01 * raw_index, -0.02, 0.0)
        poses.append({
            "raw_index": raw_index,
            "score": score,
            "branch": branch,
            "descriptor_id": adapter_case.descriptor["descriptor_id"],
            "object_from_graspgenx_row_major": pose.ravel().tolist(),
        })
    payload["proposals"] = poses
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates = _load(adapter_case, maximum_candidates=3)

    assert {row.evidence["merged_keep_stratum"] for row in candidates} == {
        "diff", "obb_top", "obb_side_0",
    }


def test_sweep_box_outside_pose_is_preserved_for_real_physics_filters(
    adapter_case,
) -> None:
    document = json.loads(adapter_case.descriptor_path.read_text(encoding="utf-8"))
    sweep = document["descriptors"][0]["graspgenx_config"]["sweep_volume"]
    sweep.update({
        "extents": [0.08, 0.08, 0.08],
        "offset": [0.0, 0.0, 0.02],
        "extents2": [0.08, 0.08, 0.08],
        "offset2": [0.0, 0.0, 0.02],
    })
    adapter_case.descriptor_path.write_text(json.dumps(document), encoding="utf-8")
    payload = copy.deepcopy(adapter_case.payload)
    payload["descriptor_manifest_sha256"] = _sha256(adapter_case.descriptor_path)
    unreachable = np.eye(4)
    unreachable[0, 3] = 2.0
    payload["proposals"].append({
        "raw_index": 3,
        "score": 1.0,
        "branch": "model",
        "descriptor_id": adapter_case.descriptor["descriptor_id"],
        "object_from_graspgenx_row_major": unreachable.ravel().tolist(),
    })
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates = _load(adapter_case)

    assert len(candidates) == 3
    assert any(row.evidence["raw_index"] == 3 for row in candidates)
    assert all("descriptor_sweep_surface_occupancy_pass" not in row.evidence for row in candidates)


def test_model_score_breaks_only_an_exact_physics_tie() -> None:
    predictions, filters, qualities = [], [], []
    for candidate_id, score in (("graspgenx_a", 0.1), ("graspgenx_z", 0.9)):
        seed = CandidateSeed(
            candidate_id, OBJECT_ID, 0, (0.0, 0.0, 0.0),
            tuple(np.eye(4).ravel()), (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), 0, score, "kcg_3f_preshape_00",
        )
        predictions.append(ClosurePrediction(seed, "CLOSURE_SURVIVE", (), (), (0.0, 0.0, 0.0), 0.01))
        filters.append(FastFilterResult(candidate_id, "FAST_SURVIVE", (), (), True, 0.01))
        qualities.append(TaskQualityResult(
            candidate_id, "TASK_SURVIVE", (1.2,), 1.2, 1.2,
            4.0, 0.3, 0.1, 0.2, 0.05,
            nominal_gravity_lift_balance_pass=True,
            nominal_parameter_task_margin=1.2,
            nominal_operation_force_cap_n=12.0,
        ))
    _research, selected, _diagnostic = select_candidate_rankings(
        tuple(predictions), tuple(filters), tuple(qualities), top_k=2,
        path_clearance_by_id={row.candidate_id: 0.01 for row in filters},
    )
    assert [row.prediction.seed.candidate_id for row in selected] == [
        "graspgenx_z", "graspgenx_a",
    ]
    assert all(row.offline_task_gate_passed is False for row in selected)


def test_nominal_task_candidate_does_not_become_formal_or_executable() -> None:
    seed = CandidateSeed(
        "nominal_only", OBJECT_ID, 0, (0.0, 0.0, 0.0),
        tuple(np.eye(4).ravel()), (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), 0, 0.9, "kcg_3f_preshape_00",
    )
    prediction = ClosurePrediction(
        seed, "CLOSURE_SURVIVE", (), (), (0.0, 0.0, 0.0), 0.01
    )
    fast_filter = FastFilterResult(
        seed.candidate_id, "FAST_SURVIVE", (), (), True, 0.01
    )
    quality = TaskQualityResult(
        seed.candidate_id, "TASK_REJECT", (0.8,), 0.8, 0.8,
        None, None, None, None, 0.0,
        nominal_gravity_lift_balance_pass=True,
        nominal_parameter_task_margin=0.8,
        nominal_operation_force_cap_n=12.0,
    )
    research, formal, diagnostic = select_candidate_rankings(
        (prediction,), (fast_filter,), (quality,), top_k=3,
    )
    assert [row.selection_status for row in research] == [
        "RESEARCH_TASK_ELIGIBLE_NOT_EXECUTABLE"
    ]
    assert research[0].offline_task_gate_passed is False
    assert formal == () and diagnostic == ()


def test_coverage_report_stays_a_failed_diagnostic_for_tiny_fixture(adapter_case) -> None:
    report = summarize_six_d_coverage(adapter_case.inputs, _load(adapter_case))
    assert report["candidate_count"] == 2
    assert report["coverage_pass"] is False
    assert report["evidence_scope"].endswith("NOT_GRASP_SUCCESS")


def test_nonrigid_or_left_handed_pose_fails_closed(adapter_case) -> None:
    for rotation in (np.diag((1.0, 1.0, -1.0)), np.diag((1.0, 1.0, 2.0))):
        payload = copy.deepcopy(adapter_case.payload)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        payload["proposals"][0]["object_from_graspgenx_row_major"] = pose.ravel().tolist()
        adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            _load(adapter_case)


def test_standardized_geometry_and_descriptor_mimic_fail_closed(adapter_case) -> None:
    with np.load(adapter_case.mesh_path, allow_pickle=False) as archive:
        vertices = np.array(archive["vertices_m"], copy=True)
        faces = np.array(archive["faces"], copy=True)
        allowed = np.array(archive["allowed_face_indices"], copy=True)
    vertices[0, 0] += 1e-4
    np.savez_compressed(
        adapter_case.mesh_path, vertices_m=vertices, faces=faces,
        allowed_face_indices=allowed,
    )
    payload = copy.deepcopy(adapter_case.payload)
    payload["standardized_mesh_sha256"] = _sha256(adapter_case.mesh_path)
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry differs"):
        _load(adapter_case)

    objects = write_standardized_object_manifest(
        {OBJECT_ID: adapter_case.inputs}, adapter_case.mesh_path.parent,
        adapter_case.mesh_path.parent / "objects-restored.json", 2048,
    )
    payload["standardized_mesh_sha256"] = objects["objects"][0]["standardized_mesh_sha256"]
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    document = json.loads(adapter_case.descriptor_path.read_text(encoding="utf-8"))
    document["descriptors"][0]["open_joint_positions_rad"]["f1j3"] += 0.01
    adapter_case.descriptor_path.write_text(json.dumps(document), encoding="utf-8")
    payload["descriptor_manifest_sha256"] = _sha256(adapter_case.descriptor_path)
    adapter_case.proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mimic"):
        _load(adapter_case)
