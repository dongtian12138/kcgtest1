"""Fail-closed full-hand sequential collision aggregation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import kcg_connector.grasp.robust.full_hand_collision as full_collision_module

from kcg_connector.grasp.robust.collision_contract import (
    CoverageMode,
    DisabledCollisionAssertion,
    build_self_collision_pair_inventory,
    load_exact_terminal_triangle_partition,
)
from kcg_connector.grasp.robust.full_hand_collision import (
    CLAIM_LIMITATIONS,
    CONTACT_RANGE_POLICY_CLAIM_LIMITATIONS,
    CONTACT_RANGE_POLICY_METHOD_ID,
    FIXED_ARM_POLICY_EMBEDDING_METHOD_ID,
    PAD_SURFACE_BLOCKER_PREFIX,
    TERMINAL_DUAL_SURFACE_COLLISION_DOMAIN,
    ContactRangePolicyCollisionCertificate,
    FullHandClosureCollisionState,
    FullHandCollisionError,
    HashBoundLinkSurface,
    HashBoundObjectSurface,
    SequentialClosureSegment,
    TerminalForbiddenSurface,
    build_fixed_arm_policy_embedding,
    certify_full_hand_contact_range_policy_closure,
    certify_full_hand_sequential_closure,
    triangle_surface_geometry_sha256,
)
from kcg_connector.grasp.robust.continuous_collision import (
    ContinuousCollisionState,
    prepare_static_triangle_surface,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.hand_contract import (
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
)
from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    IMPLICIT_ROOT_FEATURE_TYPE,
    IMPLICIT_ROOT_METHOD_ID,
    CertifiedImplicitRoot,
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalBounds,
    IntervalTransverseRootCertificate,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
)
from kcg_connector.grasp.robust.interval_policy_margin import (
    IntervalPolicyMarginState,
)
from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    ObjectGraspModel,
    TriangleMesh,
    file_sha256,
)
from kcg_connector.grasp.robust.post_generation_ranker import (
    COMPLETE_CLEARANCE_SCOPE,
    COMPLETE_POLICY_CLEARANCE_METHOD_ID,
    CandidateEvaluationState,
    CompleteContactRangeTrajectoryCollisionCertificate,
    PostGenerationRankOnlyPipeline,
    SCENARIO_COUNT,
    SCENARIO_DESIGN_SHA256,
    SCENARIO_DIMENSION,
    SCENARIO_SOBOL_SEED,
)
from kcg_connector.grasp.robust.ray_closure import (
    CLAIM_LIMITATIONS as RAY_CLOSURE_CLAIM_LIMITATIONS,
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_FOCUS_METHOD,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CLOSURE_SUFFIX_DOMINANCE_ARGUMENT,
    CertifiedContactFeatureRoot,
    CertifiedSequentialClosurePolicy,
    DISPLAY_APPROXIMATION_ROLE,
    DISTANCE_BVH_RULE,
    FEATURE_ROOT_POLICY,
    INTERNAL_FORCE_ROLE,
    INTERVAL_RULE,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_BINDING_UNBOUND_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX,
    POSSIBLE_EARLIEST_ORDERING_POLICY,
    POSSIBLE_FIRST_CONTACT_SET_METHOD_ID,
    PossibleFirstContactSet,
    RAY_EVALUATION_POLICY,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    TRAJECTORY_CLEARANCE_ROLE,
    WITNESS_RULE,
    WHOLE_PATH_SPHERE_SCREEN_RULE,
    PadClosureAudit,
    RayClosureAudit,
    RayClosureEvaluation,
    _PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE,
    _PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH,
    _PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH,
    _WHOLE_PATH_SPHERE_SEGMENT_COUNT,
    _canonical_json,
    _float64_array_hex,
    _hand_model_manifest,
)
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS,
    CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS,
    CONTACT_RANGE_POLICY_WRENCH_JOINT_DOMAIN_RULE,
    CONTACT_RANGE_POLICY_WRENCH_METHOD_ID,
    CONTACT_RANGE_POLICY_WRENCH_PARAMETRIC_CLAIM_LIMITATIONS,
    CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE,
    CONTACT_RANGE_POLICY_WRENCH_REMAINING_BLOCKERS,
    CONTACT_RANGE_POLICY_WRENCH_ROOT_DOMAIN_RULE,
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
    ContactRangePadWrenchDomain,
    ContactRangePolicyWrenchCertificate,
    ContactRangePolicyWrenchState,
    TaskWrenchEvaluationError,
    TaskWrenchEvaluator,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    AttemptStatus,
    CandidateAttemptAudit,
    CandidateLane,
    CandidateLineage,
    StaticV9AcceptedPolicy,
    TopLevelGenerationResult,
    UniqueV9Evaluation,
    V9InvocationAuditBinding,
    METHOD_ID as TOP_LEVEL_GENERATOR_METHOD_ID,
    canonicalize_v9_parameters,
)


def _backend(*, include_preshape: bool = False) -> DirectedIntervalKinematics:
    joints = {}
    pads = {}
    finger_joints = {}
    if include_preshape:
        joints["joint_preshape_a"] = JointSpec(
            name="joint_preshape_a",
            joint_type="prismatic",
            parent_link="hand_base",
            child_link="link_preshape_a",
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            limit=JointLimit(0.0, 1.0, effort=100.0),
        )
    for index, name in enumerate(("a", "b", "c")):
        joint_name = f"joint_{name}"
        link_name = f"link_{name}"
        finger_name = f"finger_{name}"
        pad_name = f"pad_{name}"
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type="prismatic",
            parent_link=(
                "link_preshape_a"
                if include_preshape and name == "a"
                else "hand_base"
            ),
            child_link=link_name,
            origin_xyz_m=(0.0, 10.0 * index, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            limit=JointLimit(0.0, 1.0, effort=100.0),
        )
        pads[pad_name] = PadGeometry(
            name=pad_name,
            finger_name=finger_name,
            link_name=link_name,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (1.0, 1.0, 1.0)),
            normal_force_capacity_n=1.0,
        )
        finger_joints[finger_name] = (
            ("joint_preshape_a", joint_name)
            if include_preshape and name == "a"
            else (joint_name,)
        )
    hand = ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names=finger_joints,
        pads=pads,
    )
    return DirectedIntervalKinematics(
        hand,
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )


def _source_triangles() -> np.ndarray:
    return np.asarray(
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, 0.0, 1.0)),
        ),
        dtype=np.float64,
    )


def _task_object_triangles() -> np.ndarray:
    return np.asarray(
        (
            (
                (100.0, 0.0, 0.0),
                (100.0, 1.0, 0.0),
                (100.0, 0.0, 1.0),
            ),
        ),
        dtype=np.float64,
    )


def _task_object_model() -> ObjectGraspModel:
    triangles = _task_object_triangles()
    mesh = TriangleMesh(
        vertices_m=triangles[0],
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        face_semantics=("external",),
    )
    return ObjectGraspModel(
        mesh=mesh,
        provenance=AssetProvenance(
            source_path="synthetic_policy_wrench_fixture.stl",
            source_sha256=hashlib.sha256(
                b"synthetic policy wrench fixture"
            ).hexdigest(),
            source_class="SYNTHETIC_ANALYTIC_TEST_FIXTURE",
            source_format="ASCII_STL",
            source_unit="m",
            meters_per_source_unit=1.0,
        ),
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        mass_kg=1.0,
        center_of_mass_m=np.asarray((100.0, 0.25, 0.25)),
        inertia_kg_m2=np.diag((0.01, 0.01, 0.01)),
        allowed_contact_semantics=frozenset(("external",)),
    )


def _local_transform() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transform[:3, 3] = (0.25, 0.5, 0.75)
    return transform


def _transform_triangles(
    triangles: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    return triangles @ transform[:3, :3].T + transform[:3, 3]


def _write_ascii_stl(path: Path, triangles: np.ndarray) -> None:
    lines = ["solid full_hand_fixture"]
    for triangle in triangles:
        normal = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        normal /= np.linalg.norm(normal)
        lines.append(
            f"  facet normal {normal[0]:.17g} "
            f"{normal[1]:.17g} {normal[2]:.17g}"
        )
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(
                f"      vertex {vertex[0]:.17g} "
                f"{vertex[1]:.17g} {vertex[2]:.17g}"
            )
        lines.extend(("    endloop", "  endfacet"))
    lines.append("endsolid full_hand_fixture")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _terminal_input(
    tmp_path: Path,
    *,
    link_name: str,
) -> tuple[HashBoundLinkSurface, TerminalForbiddenSurface]:
    source = _source_triangles()
    source_path = tmp_path / f"{link_name}.stl"
    _write_ascii_stl(source_path, source)
    pad_path = tmp_path / f"{link_name}_pad.npz"
    np.savez(
        pad_path,
        points_local_m=source[0],
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    transform = _local_transform()
    partition = load_exact_terminal_triangle_partition(
        asset_id=f"asset_{link_name}",
        link_name=link_name,
        source_stl_path=source_path,
        source_stl_sha256=file_sha256(source_path),
        source_unit="m",
        source_coverage_mode=CoverageMode.SYNTHETIC_ARRAY_FIXTURE,
        pad_npz_path=pad_path,
        pad_npz_sha256=file_sha256(pad_path),
        local_transform=transform,
    )
    full_triangles = _transform_triangles(source, transform)
    forbidden_triangles = _transform_triangles(source[1:], transform)
    full = HashBoundLinkSurface(
        link_name=link_name,
        source_asset_sha256=file_sha256(source_path),
        geometry_sha256=triangle_surface_geometry_sha256(full_triangles),
        triangles_link_m=full_triangles,
    )
    forbidden = HashBoundLinkSurface(
        link_name=link_name,
        source_asset_sha256=file_sha256(source_path),
        geometry_sha256=triangle_surface_geometry_sha256(
            forbidden_triangles
        ),
        triangles_link_m=forbidden_triangles,
    )
    return full, TerminalForbiddenSurface(
        link_name=link_name,
        partition=partition,
        nonpad_forbidden_surface=forbidden,
    )


def _pad_audit(
    pad_name: str,
    finger_name: str,
    phase: float,
) -> PadClosureAudit:
    return PadClosureAudit(
        pad_name=pad_name,
        finger_name=finger_name,
        verified_triangle_count=1,
        witness_count=3,
        exact_fk_interval_evaluations=1,
        leading_witness_evaluations=1,
        first_hit_rays_cast=0,
        finite_chord_feature_candidates=0,
        nonlinear_feature_roots_solved=0,
        nonlinear_root_fk_evaluations=0,
        distance_bvh_node_visits=1,
        distance_triangle_tests=1,
        certified_free_interval_count=1,
        certified_witness_path_clearance_lower_bound_m=0.0,
        interval_point_motion_evaluations=1,
        swept_face_candidate_count=1,
        interval_pair_evaluation_count=1,
        certified_contact_root_count=1,
        unresolved_witness_face_pair_count=0,
        cofirst_root_count=1,
        competing_root_order_block_count=0,
        acceptance_ray_call_count=0,
        selected_triangle_index=0,
        selected_witness_index=0,
        selected_object_face_index=0,
        selected_normalized_closure=phase,
        selected_closure_interval_width=(
            np.nextafter(phase, np.inf)
            - np.nextafter(phase, -np.inf)
        ),
        selected_spatial_error_bound_m=0.0,
        selected_root_phase_lower=float(
            np.nextafter(phase, -np.inf)
        ),
        selected_root_phase_upper=float(
            np.nextafter(phase, np.inf)
        ),
        selected_pad_approach_lower=1.0,
        selected_path_local_free_side_approach_lower=1.0,
        selected_object_source_winding_free_side_sign=1,
        first_contact_classification=(
            "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
        ),
    )


def _runtime_pad_hash(
    points: np.ndarray,
    faces: np.ndarray,
) -> str:
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
    return digest.hexdigest()


def _v9_evaluation(
    *,
    backend: DirectedIntervalKinematics,
    terminals: tuple[TerminalForbiddenSurface, ...],
    object_surface: HashBoundObjectSurface,
    segments: tuple[SequentialClosureSegment, ...],
) -> RayClosureEvaluation:
    phases = tuple(segment.phase.upper for segment in segments)
    pad_names = tuple(segment.pad_name for segment in segments)
    contacts = tuple(
        PlannedPadContact(
            pad_name=pad_name,
            position_object_m=(100.0, float(index), 0.0),
            path_local_free_side_normal_object=(-1.0, 0.0, 0.0),
            surface_coordinates=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, phase),
        )
        for index, (pad_name, phase) in enumerate(zip(pad_names, phases))
    )
    final_joint_positions = np.asarray(
        segments[-1].q_start,
        dtype=np.float64,
    ) + segments[-1].phase.upper * np.asarray(
        segments[-1].direction,
        dtype=np.float64,
    )
    candidate = GraspCandidate.from_matrix(
        object_from_hand=np.eye(4),
        independent_joint_positions_rad=final_joint_positions,
        planned_pad_contacts=contacts,
        internal_normal_forces_n=(0.0, 0.0, 0.0),
    )
    pad_audits = tuple(
        _pad_audit(pad_name, f"finger_{name}", phase)
        for pad_name, name, phase in zip(
            pad_names, ("a", "b", "c"), phases
        )
    )
    supports = tuple(
        tuple(
            name
            for name, value in zip(
                backend.hand_model.independent_joint_names,
                segment.direction,
            )
            if value != 0.0
        )
        for segment in segments
    )
    physical_directions = tuple(segment.direction for segment in segments)
    support_names = {name for row in supports for name in row}
    lower_limits, upper_limits = backend.hand_model.joint_limit_vectors()
    preshape_joint_names = tuple(
        name
        for index, name in enumerate(
            backend.hand_model.independent_joint_names
        )
        if name not in support_names
        and upper_limits[index] > lower_limits[index]
    )
    parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
        f"preshape_joint_unit:{name}" for name in preshape_joint_names
    )
    closure_open_joint_positions = tuple(
        float(
            lower_limits[index]
            if direction[index] > 0.0
            else upper_limits[index]
        )
        for direction in physical_directions
        for index in [
            next(
                joint_index
                for joint_index, value in enumerate(direction)
                if value != 0.0
            )
        ]
    )
    pad_geometry_hashes: list[str] = []
    pad_runtime_hashes: list[str] = []
    pad_links: list[str] = []
    pad_manifest: list[dict[str, object]] = []
    terminal_by_link = {row.link_name: row for row in terminals}
    for segment in segments:
        terminal = terminal_by_link[segment.active_link_name]
        with np.load(
            terminal.partition.pad_source_path,
            allow_pickle=False,
        ) as arrays:
            points = np.asarray(arrays["points_local_m"], dtype=np.float64)
            faces = np.asarray(arrays["faces"], dtype=np.int64)
        runtime_hash = _runtime_pad_hash(points, faces)
        hand_pad = backend.hand_model.pads[segment.pad_name]
        pad_geometry_hashes.append(terminal.partition.pad_source_sha256)
        pad_runtime_hashes.append(runtime_hash)
        pad_links.append(segment.active_link_name)
        pad_manifest.append(
            {
                "name": segment.pad_name,
                "finger_name": hand_pad.finger_name,
                "link_name": segment.active_link_name,
                "origin_xyz_m": _float64_array_hex(
                    hand_pad.origin_xyz_m
                ),
                "origin_rpy_rad": _float64_array_hex(
                    hand_pad.origin_rpy_rad
                ),
                "coordinate_frame": segment.active_link_name,
                "unit": "m",
                "normal_force_capacity_n": float(1.0).hex(),
                "source_mesh_repository_relative_path": (
                    f"synthetic/{segment.pad_name}.npz"
                ),
                "source_mesh_sha256": (
                    terminal.partition.pad_source_sha256
                ),
                "source_mesh_byte_count": (
                    terminal.partition.pad_source_path.stat().st_size
                ),
                "runtime_geometry_sha256": runtime_hash,
                "vertex_count": len(points),
                "triangle_count": len(faces),
            }
        )
    model_document = {
        "schema": MODEL_CONTRACT_DIGEST_METHOD_ID,
        "object": {
            "geometry_sha256": (
                object_surface.ray_closure_object_geometry_sha256
            ),
            "assembly_axis": _float64_array_hex((0.0, 0.0, 1.0)),
            "assembly_axis_origin_m": _float64_array_hex(
                (0.0, 0.0, 0.0)
            ),
        },
        "task_frame": {
            "source": "SYNTHETIC_PRE_REGISTERED_TASK_FRAME",
            "pre_registered_transverse_axis_object": (
                _float64_array_hex((1.0, 0.0, 0.0))
            ),
            "basis_object": _float64_array_hex(np.eye(3)),
        },
        "hand": _hand_model_manifest(backend.hand_model),
        "verified_pads": pad_manifest,
        "closure": {
            "closing_directions_unit": _float64_array_hex(
                physical_directions
            ),
            "closing_directions_physical": _float64_array_hex(
                physical_directions
            ),
            "independent_actuation_supports": [
                list(row) for row in supports
            ],
            "parameter_layout": list(parameter_layout),
        },
        "ray_closure": {
            "method_id": RAY_CLOSURE_METHOD_ID,
            "closure_parameter_domain_id": CLOSURE_PARAMETER_DOMAIN_ID,
            "closure_focus_method": CLOSURE_FOCUS_METHOD,
            "feature_root_policy": FEATURE_ROOT_POLICY,
            "possible_first_contact_set_method_id": (
                POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
            ),
            "possible_earliest_ordering_policy": (
                POSSIBLE_EARLIEST_ORDERING_POLICY
            ),
            "representative_proposal_role": (
                CANDIDATE_REPRESENTATIVE_ROLE
            ),
            "display_approximation_role": DISPLAY_APPROXIMATION_ROLE,
            "ray_evaluation_policy": RAY_EVALUATION_POLICY,
            "witness_rule": WITNESS_RULE,
            "interval_rule": INTERVAL_RULE,
            "whole_path_sphere_screen_rule": (
                WHOLE_PATH_SPHERE_SCREEN_RULE
            ),
            "whole_path_sphere_segment_count": (
                _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            ),
            "pad_surface_sphere_hierarchy_maximum_depth": (
                _PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH
            ),
            "pad_surface_aabb_maximum_temporal_refinement_depth": (
                _PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH
            ),
            "pad_surface_aabb_maximum_moving_triangle_pair_tests_per_coverage": (
                _PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE
            ),
            "object_contact_normal_policy": (
                OBJECT_CONTACT_NORMAL_POLICY
            ),
            "pad_surface_normal_policy": PAD_SURFACE_NORMAL_POLICY,
            "maximum_subdivision_intervals": 100,
        },
        "interval_backend": {
            "method_id": INTERVAL_KINEMATICS_METHOD_ID,
            "decimal_precision": backend.options.decimal_precision,
            "maximum_root_bisection_iterations": (
                backend.options.maximum_root_bisection_iterations
            ),
        },
    }
    canonical_model_json = _canonical_json(model_document)
    model_contract_sha256 = hashlib.sha256(
        canonical_model_json.encode("utf-8")
    ).hexdigest()
    audit = RayClosureAudit(
        method_id=RAY_CLOSURE_METHOD_ID,
        numerical_policy="fixture",
        witness_rule=WITNESS_RULE,
        interval_rule=INTERVAL_RULE,
        distance_bvh_rule=DISTANCE_BVH_RULE,
        ray_evaluation_policy=RAY_EVALUATION_POLICY,
        feature_root_policy=FEATURE_ROOT_POLICY,
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        parameter_layout=parameter_layout,
        pad_order=pad_names,
        full_verified_pad_mesh_used=True,
        pad_face_subset_input_allowed=False,
        independent_actuation_supports=supports,
        closure_parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        closure_suffix_dominance_argument=(
            CLOSURE_SUFFIX_DOMINANCE_ARGUMENT
        ),
        preshape_joint_names=preshape_joint_names,
        closure_open_joint_positions_rad=closure_open_joint_positions,
        maximum_subdivision_intervals=100,
        interval_arithmetic_method_id=INTERVAL_KINEMATICS_METHOD_ID,
        interval_decimal_precision=backend.options.decimal_precision,
        maximum_root_bisection_iterations=(
            backend.options.maximum_root_bisection_iterations
        ),
        subdivision_intervals_used=3,
        subdivision_budget_exhausted=False,
        internal_force_role=INTERNAL_FORCE_ROLE,
        trajectory_clearance_m=0.0,
        trajectory_clearance_role=TRAJECTORY_CLEARANCE_ROLE,
        task_frame_source="SYNTHETIC_PRE_REGISTERED_TASK_FRAME",
        closure_focus_method=CLOSURE_FOCUS_METHOD,
        distance_bvh_node_count=1,
        pad_audits=pad_audits,
        claim_limitations=RAY_CLOSURE_CLAIM_LIMITATIONS,
        failure_reason=None,
        model_binding_complete=True,
        model_binding_status=MODEL_BINDING_COMPLETE_STATUS,
        object_geometry_sha256=(
            object_surface.ray_closure_object_geometry_sha256
        ),
        model_contract_sha256=model_contract_sha256,
        pad_geometry_sha256=tuple(pad_geometry_hashes),
        pad_runtime_geometry_sha256=tuple(pad_runtime_hashes),
        pad_link_names=tuple(pad_links),
        closing_directions_physical=physical_directions,
        model_contract_canonical_json=canonical_model_json,
    )
    return RayClosureEvaluation(candidate=candidate, audit=audit)


def _replace_bound_model_document(
    audit: RayClosureAudit,
    document: dict[str, object],
    **audit_changes: object,
) -> RayClosureAudit:
    canonical_json = _canonical_json(document)
    return replace(
        audit,
        model_contract_canonical_json=canonical_json,
        model_contract_sha256=hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest(),
        **audit_changes,
    )


def _segments(
    maximum_subdivision_intervals: int = 6,
    *,
    include_preshape: bool = False,
) -> tuple[SequentialClosureSegment, ...]:
    if include_preshape:
        q0 = (0.5, 0.0, 0.0, 0.0)
        q1 = (0.5, 0.1, 0.0, 0.0)
        q2 = (0.5, 0.1, 0.2, 0.0)
        directions = (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    else:
        q0 = (0.0, 0.0, 0.0)
        q1 = (0.1, 0.0, 0.0)
        q2 = (0.1, 0.2, 0.0)
        directions = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    return (
        SequentialClosureSegment(
            segment_index=0,
            pad_name="pad_a",
            active_link_name="link_a",
            q_start=q0,
            direction=directions[0],
            phase=IntervalBounds(0.0, 0.1),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
        SequentialClosureSegment(
            segment_index=1,
            pad_name="pad_b",
            active_link_name="link_b",
            q_start=q1,
            direction=directions[1],
            phase=IntervalBounds(0.0, 0.2),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
        SequentialClosureSegment(
            segment_index=2,
            pad_name="pad_c",
            active_link_name="link_c",
            q_start=q2,
            direction=directions[2],
            phase=IntervalBounds(0.0, 0.3),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    budget: int = 6,
    include_preshape: bool = False,
):
    backend = _backend(include_preshape=include_preshape)
    terminal_rows = tuple(
        _terminal_input(tmp_path, link_name=f"link_{name}")
        for name in ("a", "b", "c")
    )
    links = tuple(row[0] for row in terminal_rows)
    terminals = tuple(row[1] for row in terminal_rows)
    object_triangles = _task_object_triangles()
    object_surface = HashBoundObjectSurface(
        object_id="object_fixture",
        source_asset_sha256=hashlib.sha256(b"object fixture").hexdigest(),
        geometry_sha256=triangle_surface_geometry_sha256(
            object_triangles
        ),
        ray_closure_object_geometry_sha256=(
            _task_object_model().geometry_sha256
        ),
        triangles_object_m=object_triangles,
    )
    inventory = build_self_collision_pair_inventory(
        link_names=("link_a", "link_b", "link_c"),
        srdf_assertions=(
            DisabledCollisionAssertion("link_a", "link_b", "Adjacent"),
            DisabledCollisionAssertion("link_b", "link_c", "Never"),
        ),
    )
    segments = _segments(
        budget,
        include_preshape=include_preshape,
    )
    evaluation = _v9_evaluation(
        backend=backend,
        terminals=terminals,
        object_surface=object_surface,
        segments=segments,
    )
    return (
        backend,
        links,
        terminals,
        object_surface,
        inventory,
        evaluation,
        segments,
    )


def _possible_contact_set(
    pad_name: str,
    ordinal: int,
    phase: float,
) -> PossibleFirstContactSet:
    lower = float(np.nextafter(phase, -np.inf))
    upper = float(np.nextafter(phase, np.inf))
    implicit = CertifiedImplicitRoot(
        method_id=IMPLICIT_ROOT_METHOD_ID,
        equation_sha256=hashlib.sha256(
            f"equation:{pad_name}".encode("utf-8")
        ).hexdigest(),
        feature_identity_sha256=hashlib.sha256(
            f"feature:{pad_name}".encode("utf-8")
        ).hexdigest(),
        feature_type=IMPLICIT_ROOT_FEATURE_TYPE,
        isolating_interval=IntervalBounds(lower, upper),
        value_at_lower=IntervalBounds(0.1, 0.2),
        value_at_upper=IntervalBounds(-0.2, -0.1),
        derivative=IntervalBounds(-2.0, -1.0),
        uniqueness_proven=True,
        display_approximation=phase,
        display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
    )
    center_y = 0.2 + 0.2 * float(ordinal)
    certificate = IntervalTransverseRootCertificate(
        implicit_root=implicit,
        triangle_edge_halfspaces=(
            IntervalBounds(0.1, 0.2),
            IntervalBounds(0.1, 0.2),
            IntervalBounds(0.1, 0.2),
        ),
        pad_approach=IntervalBounds(0.5, 1.0),
        path_local_free_side_approach=IntervalBounds(0.5, 1.0),
        object_source_winding_free_side_sign=1,
        position_object_m=(
            IntervalBounds(100.0, 100.0),
            IntervalBounds(center_y - 0.01, center_y + 0.01),
            IntervalBounds(0.19, 0.21),
        ),
        bisection_iterations=8,
        method_id=INTERVAL_KINEMATICS_METHOD_ID,
        decimal_precision=80,
    )
    root = CertifiedContactFeatureRoot(
        pad_name=pad_name,
        witness_flat_index=ordinal,
        pad_triangle_index=0,
        witness_index=ordinal,
        object_face_index=0,
        semantic_classification=(
            "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
        ),
        certificate=certificate,
    )
    return PossibleFirstContactSet.from_certified_roots((root,))


def _policy_fixture(tmp_path: Path, *, include_preshape: bool = False):
    fixture = _fixture(tmp_path, include_preshape=include_preshape)
    backend = fixture[0]
    evaluation = fixture[5]
    segments = fixture[6]
    contact_sets = tuple(
        _possible_contact_set(
            segment.pad_name,
            index,
            segment.phase.upper,
        )
        for index, segment in enumerate(segments)
    )
    policy = CertifiedSequentialClosurePolicy(
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        initial_independent_joint_positions_rad=tuple(
            float(value) for value in segments[0].q_start
        ),
        independent_joint_names=tuple(
            backend.hand_model.independent_joint_names
        ),
        pad_order=tuple(segment.pad_name for segment in segments),
        independent_actuation_supports=(
            evaluation.audit.independent_actuation_supports
        ),
        closing_directions_physical=(
            evaluation.audit.closing_directions_physical
        ),
        possible_first_contact_sets=contact_sets,
        object_geometry_sha256=evaluation.audit.object_geometry_sha256,
        model_contract_sha256=evaluation.audit.model_contract_sha256,
    )
    policy_pad_audits = tuple(
        replace(
            pad_audit,
            possible_earliest_root_count=1,
            possible_first_contact_set_sha256=contact_set.set_sha256,
            selected_normalized_closure_role=DISPLAY_APPROXIMATION_ROLE,
        )
        for pad_audit, contact_set in zip(
            evaluation.audit.pad_audits, contact_sets
        )
    )
    policy_audit = replace(
        evaluation.audit,
        pad_audits=policy_pad_audits,
        failure_reason=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
        candidate_role=CANDIDATE_REPRESENTATIVE_ROLE,
        possible_first_contact_set_sha256=tuple(
            row.set_sha256 for row in contact_sets
        ),
    )
    return fixture, policy, policy_audit


def _ranking_policy_fixture(tmp_path: Path):
    return _policy_fixture(tmp_path, include_preshape=True)


def _certify_policy(
    policy_fixture,
    *,
    maximum_subdivision_intervals: int = 6,
) -> ContactRangePolicyCollisionCertificate:
    fixture, policy, policy_audit = policy_fixture
    return certify_full_hand_contact_range_policy_closure(
        backend=fixture[0],
        link_surfaces=fixture[1],
        terminal_forbidden_surfaces=fixture[2],
        object_surface=fixture[3],
        self_pair_inventory=fixture[4],
        sequential_closure_policy=policy,
        v9_audit=policy_audit,
        maximum_subdivision_intervals=maximum_subdivision_intervals,
    )


def _aggregate_backend_for_fixed_arm_embedding(
    source_backend: DirectedIntervalKinematics,
    *,
    altered_hand_limit: bool = False,
) -> DirectedIntervalKinematics:
    source = source_backend.hand_model
    arm_joint = JointSpec(
        name="arm_joint",
        joint_type="prismatic",
        parent_link="world",
        child_link=source.base_link,
        origin_xyz_m=(0.0, 0.0, 0.5),
        origin_rpy_rad=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        limit=JointLimit(-1.0, 1.0, effort=100.0),
    )
    joints = {"arm_joint": arm_joint, **dict(source.joints)}
    if altered_hand_limit:
        original = joints["joint_a"]
        joints["joint_a"] = replace(
            original,
            limit=JointLimit(0.0, 0.9, effort=100.0),
        )
    aggregate = ThreeFingerHandModel(
        base_link="world",
        joints=joints,
        joint_order=("arm_joint", *source.joint_order),
        finger_joint_names={
            name: chain.joint_names
            for name, chain in source.fingers.items()
        },
        pads=dict(source.pads),
    )
    return DirectedIntervalKinematics(aggregate, source_backend.options)


def test_fixed_arm_embedding_preserves_hand_policy_and_reaches_collision_gate(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    source_backend = fixture[0]
    aggregate_backend = _aggregate_backend_for_fixed_arm_embedding(
        source_backend
    )
    embedding = build_fixed_arm_policy_embedding(
        source_backend=source_backend,
        aggregate_backend=aggregate_backend,
        sequential_closure_policy=policy,
        fixed_aggregate_only_joint_positions_rad=(("arm_joint", 0.25),),
    )

    assert embedding.source_to_aggregate_indices == (1, 2, 3)
    assert all(
        row[0] == 0.0
        for row in embedding.embedded_closing_directions_physical
    )
    certificate = certify_full_hand_contact_range_policy_closure(
        backend=aggregate_backend,
        link_surfaces=fixture[1],
        terminal_forbidden_surfaces=fixture[2],
        object_surface=fixture[3],
        self_pair_inventory=fixture[4],
        sequential_closure_policy=policy,
        v9_audit=policy_audit,
        maximum_subdivision_intervals=6,
        fixed_arm_policy_embedding=embedding,
    )

    assert certificate.state is FullHandClosureCollisionState.NOT_CERTIFIABLE
    assert certificate.audit.policy_kinematic_binding_method_id == (
        FIXED_ARM_POLICY_EMBEDDING_METHOD_ID
    )
    assert certificate.audit.policy_kinematic_binding_sha256 == (
        embedding.certificate_sha256
    )
    assert certificate.audit.fixed_aggregate_joint_bindings == (
        ("arm_joint", 0.25),
    )


def test_fixed_arm_embedding_rejects_hand_submodel_or_fixed_joint_drift(
    tmp_path: Path,
) -> None:
    fixture, policy, _policy_audit = _policy_fixture(tmp_path)
    source_backend = fixture[0]
    changed_backend = _aggregate_backend_for_fixed_arm_embedding(
        source_backend,
        altered_hand_limit=True,
    )
    with pytest.raises(FullHandCollisionError, match="differ from source"):
        build_fixed_arm_policy_embedding(
            source_backend=source_backend,
            aggregate_backend=changed_backend,
            sequential_closure_policy=policy,
            fixed_aggregate_only_joint_positions_rad=(("arm_joint", 0.25),),
        )
    aggregate_backend = _aggregate_backend_for_fixed_arm_embedding(
        source_backend
    )
    with pytest.raises(FullHandCollisionError, match="absent, reordered"):
        build_fixed_arm_policy_embedding(
            source_backend=source_backend,
            aggregate_backend=aggregate_backend,
            sequential_closure_policy=policy,
            fixed_aggregate_only_joint_positions_rad=(("wrong_joint", 0.25),),
        )


_POLICY_RANK_GENERATION_SHA256 = "e" * 64


def _policy_generation_result(
    policy: CertifiedSequentialClosurePolicy,
    audit: RayClosureAudit,
) -> TopLevelGenerationResult:
    parameters = (0.1, 0.2, 0.3, 0.4, 0.5)
    canonical = canonicalize_v9_parameters(
        parameters,
        parameter_layout=audit.parameter_layout,
    )
    accepted_lineage = CandidateLineage(
        attempt_index=0,
        lane=CandidateLane.DIRECT_V9,
        lane_point_index=0,
        sobol_seed=20260820,
        sobol_parameters_unit=parameters,
        anchor_pad_name=None,
        proposal_audit=None,
        proposal_failure_reason=None,
    )
    binding = V9InvocationAuditBinding(
        method_id=RAY_CLOSURE_METHOD_ID,
        parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        parameter_layout=tuple(audit.parameter_layout),
        requested_parameters_unit=parameters,
        requested_parameter_key_hex=canonical.exact_key_hex,
        raw_v9_audit=audit,
    )
    accepted_attempt = CandidateAttemptAudit(
        lineage=accepted_lineage,
        status=AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=canonical.exact_key_hex,
        duplicate_of_attempt_index=None,
        v9_audit=audit,
        invocation_binding=binding,
        v9_failure_reason=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
        failure_reason=None,
    )
    unique = UniqueV9Evaluation(
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=canonical.exact_key_hex,
        first_attempt_index=0,
        lineage=(accepted_lineage,),
        candidate=None,
        sequential_closure_policy=policy,
        v9_audit=audit,
        invocation_binding=binding,
        status=AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
        v9_failure_reason=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    )
    accepted_policy = StaticV9AcceptedPolicy(
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=canonical.exact_key_hex,
        sequential_closure_policy=policy,
        v9_audit=audit,
        invocation_binding=binding,
        lineage=(accepted_lineage,),
    )
    attempts = [accepted_attempt]
    for index in range(1, 128):
        lineage = CandidateLineage(
            attempt_index=index,
            lane=CandidateLane.SURFACE_PAD_A,
            lane_point_index=index // 4,
            sobol_seed=20260821,
            sobol_parameters_unit=(0.0,) * 6,
            anchor_pad_name=policy.pad_order[0],
            proposal_audit=None,
            proposal_failure_reason="FIXTURE_NO_PROPOSAL",
        )
        attempts.append(
            CandidateAttemptAudit(
                lineage=lineage,
                status=AttemptStatus.PROPOSAL_REJECTED,
                v9_parameters_unit=None,
                v9_parameter_key_hex=None,
                duplicate_of_attempt_index=None,
                v9_audit=None,
                invocation_binding=None,
                v9_failure_reason=None,
                failure_reason="FIXTURE_NO_PROPOSAL",
            )
        )
    return TopLevelGenerationResult(
        method_id=TOP_LEVEL_GENERATOR_METHOD_ID,
        contract_hash_sha256=_POLICY_RANK_GENERATION_SHA256,
        total_attempt_budget=128,
        attempts_per_lane=32,
        local_refinement_evaluation_budget=0,
        attempts=tuple(attempts),
        unique_v9_evaluations=(unique,),
        accepted_candidates=(),
        accepted_policies=(accepted_policy,),
        v9_evaluation_count=1,
        duplicate_attempt_count=0,
        proposal_failure_count=127,
    )


_POLICY_WRENCH_LP_OPTIONS = LinearProgramSolverOptions.from_mapping(
    {
        "solver": "SCIPY_HIGHS",
        "constraint_scaling": "ROW_AND_COLUMN_INF_NORM",
        "maximum_iterations": 10000,
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
        "ipm_optimality_tolerance": 1.0e-10,
        "physical_acceptance_gate": False,
    }
)


def _policy_wrench_evaluator() -> TaskWrenchEvaluator:
    return TaskWrenchEvaluator(
        object_model=_task_object_model(),
        characteristic_radius_m=0.1,
        friction_coefficient_interval=(0.3, 0.6),
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        gravity_direction_object=(0.0, 0.0, -1.0),
        task_frame_rotation_object=np.eye(3),
        gravity_acceleration_m_s2=9.81,
        lift_acceleration_m_s2=0.0,
        maximum_inner_approximation_relative_error=0.1,
        cone_edge_multiplier=1,
        solver_options=_POLICY_WRENCH_LP_OPTIONS,
    )


def _evaluate_policy_wrench(policy_fixture):
    fixture, policy, policy_audit = policy_fixture
    collision_certificate = _certify_policy(policy_fixture)
    evaluator = _policy_wrench_evaluator()
    certificate = evaluator.evaluate_contact_range_policy(
        policy,
        np.asarray(((0.0,), (0.5,), (1.0,)), dtype=np.float64),
        v9_audit=policy_audit,
        hand_model=fixture[0].hand_model,
        policy_collision_certificate=collision_certificate,
    )
    return evaluator, collision_certificate, certificate


def _certify(fixture):
    (
        backend,
        links,
        terminals,
        object_surface,
        inventory,
        evaluation,
        segments,
    ) = fixture
    return certify_full_hand_sequential_closure(
        backend=backend,
        link_surfaces=links,
        terminal_forbidden_surfaces=terminals,
        object_surface=object_surface,
        self_pair_inventory=inventory,
        v9_evaluation=evaluation,
        segments=segments,
    )


def test_all_checkable_pairs_free_still_remains_not_certifiable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    certificate = _certify(fixture)

    assert certificate.state is (
        FullHandClosureCollisionState.NOT_CERTIFIABLE
    )
    assert certificate.audit.checkable_collision_gates_passed
    assert certificate.audit.all_link_object_domains_covered
    assert certificate.audit.all_self_pair_domains_covered
    assert certificate.audit.evaluated_link_object_domain_count == 9
    assert certificate.audit.certified_free_link_object_domain_count == 9
    assert certificate.audit.evaluated_self_pair_domain_count == 9
    assert certificate.audit.certified_free_self_pair_domain_count == 9
    assert certificate.audit.segment_budget_usage == (
        (6, 6, 0),
        (6, 6, 0),
        (6, 6, 0),
    )
    assert certificate.audit.srdf_exemptions_applied is False
    assert certificate.audit.claim_limitations == CLAIM_LIMITATIONS
    assert certificate.audit.ray_closure_object_geometry_sha256 == (
        fixture[3].ray_closure_object_geometry_sha256
    )
    assert certificate.audit.ray_model_contract_sha256 == (
        fixture[5].audit.model_contract_sha256
    )
    assert not any(
        row.startswith("V9_EVIDENCE_NOT_ASSET")
        for row in certificate.audit.blockers
    )
    assert not any(
        row.startswith("V9_AUDIT_DOES_NOT_BIND")
        for row in certificate.audit.claim_limitations
    )
    assert all(
        row.certificate.audit.all_processed_pairs_accounted_for
        for row in certificate.link_object_certificates
        + certificate.self_pair_certificates
    )
    for pad_name, link_name in (
        ("pad_a", "link_a"),
        ("pad_b", "link_b"),
        ("pad_c", "link_c"),
    ):
        assert (
            f"{PAD_SURFACE_BLOCKER_PREFIX}:{pad_name}:{link_name}"
            in certificate.audit.blockers
        )
    assert (
        "V9_REPRESENTATIVE_PHASE_IS_ROOT_BRACKET_MIDPOINT_"
        "NOT_EXACT_CONTACT_ENDPOINT"
        in certificate.audit.blockers
    )
    assert (
        "SOLID_CONTAINMENT_OR_INITIAL_OUTSIDE_CERTIFICATE_UNAVAILABLE"
        in certificate.audit.blockers
    )
    with pytest.raises(FrozenInstanceError):
        certificate.state = FullHandClosureCollisionState.CERTIFIED
    with pytest.raises(
        FullHandCollisionError,
        match="certified full-hand state lacks complete evidence",
    ):
        replace(
            certificate,
            state=FullHandClosureCollisionState.CERTIFIED,
        )


def test_unbound_synthetic_v9_audit_fails_closed(tmp_path: Path) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    unbound_audit = replace(
        evaluation.audit,
        model_binding_complete=False,
        model_binding_status=MODEL_BINDING_UNBOUND_STATUS,
        object_geometry_sha256="UNBOUND_SYNTHETIC",
        model_contract_sha256="UNBOUND_SYNTHETIC",
        pad_geometry_sha256=(),
        pad_runtime_geometry_sha256=(),
        pad_link_names=(),
        closing_directions_physical=(),
        model_contract_canonical_json="",
    )
    fixture[5] = replace(evaluation, audit=unbound_audit)

    with pytest.raises(
        FullHandCollisionError,
        match="synthetic, unbound, or incomplete",
    ):
        _certify(tuple(fixture))


def test_nonproduction_model_binding_status_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    object.__setattr__(
        evaluation.audit,
        "model_binding_status",
        "FORGED_NONPRODUCTION_STATUS",
    )

    with pytest.raises(
        FullHandCollisionError,
        match="synthetic, unbound, or incomplete",
    ):
        _certify(tuple(fixture))


def test_model_contract_canonical_json_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    object.__setattr__(
        evaluation.audit,
        "model_contract_canonical_json",
        evaluation.audit.model_contract_canonical_json + " ",
    )

    with pytest.raises(
        FullHandCollisionError,
        match="canonical JSON digest does not recompute",
    ):
        _certify(tuple(fixture))


def test_self_consistent_object_hash_tamper_fails_explicit_binding(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    document = json.loads(evaluation.audit.model_contract_canonical_json)
    tampered_hash = "f" * 64
    document["object"]["geometry_sha256"] = tampered_hash
    tampered_audit = _replace_bound_model_document(
        evaluation.audit,
        document,
        object_geometry_sha256=tampered_hash,
    )
    fixture[5] = replace(evaluation, audit=tampered_audit)

    with pytest.raises(
        FullHandCollisionError,
        match="object geometry differs from the hash-bound object surface",
    ):
        _certify(tuple(fixture))


def test_self_consistent_hand_manifest_tamper_fails_backend_binding(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    document = json.loads(evaluation.audit.model_contract_canonical_json)
    document["hand"]["base_link"] = "forged_hand_base"
    tampered_audit = _replace_bound_model_document(
        evaluation.audit,
        document,
    )
    fixture[5] = replace(evaluation, audit=tampered_audit)

    with pytest.raises(
        FullHandCollisionError,
        match="object or complete hand binding differs",
    ):
        _certify(tuple(fixture))


@pytest.mark.parametrize(
    ("document_field", "audit_field", "tampered_value"),
    (
        ("source_mesh_sha256", "pad_geometry_sha256", "d" * 64),
        (
            "runtime_geometry_sha256",
            "pad_runtime_geometry_sha256",
            "e" * 64,
        ),
        ("link_name", "pad_link_names", "forged_link"),
    ),
)
def test_self_consistent_pad_evidence_tamper_fails_terminal_binding(
    tmp_path: Path,
    document_field: str,
    audit_field: str,
    tampered_value: str,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    document = json.loads(evaluation.audit.model_contract_canonical_json)
    document["verified_pads"][0][document_field] = tampered_value
    audit_values = list(getattr(evaluation.audit, audit_field))
    audit_values[0] = tampered_value
    tampered_audit = _replace_bound_model_document(
        evaluation.audit,
        document,
        **{audit_field: tuple(audit_values)},
    )
    fixture[5] = replace(evaluation, audit=tampered_audit)

    with pytest.raises(
        FullHandCollisionError,
        match="PAD source/runtime/link binding differs at row 0",
    ):
        _certify(tuple(fixture))


def test_one_ulp_direction_tamper_fails_exact_path_binding(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    evaluation = fixture[5]
    document = json.loads(evaluation.audit.model_contract_canonical_json)
    directions = [
        list(row) for row in evaluation.audit.closing_directions_physical
    ]
    directions[0][0] = float(np.nextafter(1.0, 0.0))
    document["closure"]["closing_directions_physical"] = (
        _float64_array_hex(directions)
    )
    tampered_audit = _replace_bound_model_document(
        evaluation.audit,
        document,
        closing_directions_physical=tuple(
            tuple(row) for row in directions
        ),
    )
    fixture[5] = replace(evaluation, audit=tampered_audit)

    with pytest.raises(
        FullHandCollisionError,
        match="supports/directions differ from the explicit path",
    ):
        _certify(tuple(fixture))


def test_nonidentity_transform_and_face_permutations_preserve_result(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    links = tuple(
        HashBoundLinkSurface(
            link_name=row.link_name,
            source_asset_sha256=row.source_asset_sha256,
            geometry_sha256=row.geometry_sha256,
            triangles_link_m=row.triangles_link_m[::-1, ::-1],
        )
        for row in fixture[1]
    )
    terminals = tuple(
        TerminalForbiddenSurface(
            link_name=row.link_name,
            partition=row.partition,
            nonpad_forbidden_surface=HashBoundLinkSurface(
                link_name=row.link_name,
                source_asset_sha256=(
                    row.nonpad_forbidden_surface.source_asset_sha256
                ),
                geometry_sha256=(
                    row.nonpad_forbidden_surface.geometry_sha256
                ),
                triangles_link_m=(
                    row.nonpad_forbidden_surface.triangles_link_m[:, ::-1]
                ),
            ),
        )
        for row in fixture[2]
    )
    fixture[1] = links
    fixture[2] = terminals

    certificate = _certify(tuple(fixture))

    assert certificate.audit.checkable_collision_gates_passed
    assert certificate.state is (
        FullHandClosureCollisionState.NOT_CERTIFIABLE
    )


def test_terminal_forbidden_subset_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    first = fixture[2][0]
    wrong_triangles = fixture[1][0].triangles_link_m[:1]
    wrong_surface = HashBoundLinkSurface(
        link_name=first.link_name,
        source_asset_sha256=(
            first.nonpad_forbidden_surface.source_asset_sha256
        ),
        geometry_sha256=triangle_surface_geometry_sha256(wrong_triangles),
        triangles_link_m=wrong_triangles,
    )
    fixture[2] = (
        TerminalForbiddenSurface(
            link_name=first.link_name,
            partition=first.partition,
            nonpad_forbidden_surface=wrong_surface,
        ),
        *fixture[2][1:],
    )

    with pytest.raises(
        FullHandCollisionError,
        match="explicit terminal non-PAD surface differs",
    ):
        _certify(tuple(fixture))


def test_shared_segment_budget_never_multiplies_by_domain_count(
    tmp_path: Path,
) -> None:
    certificate = _certify(_fixture(tmp_path, budget=1))

    assert certificate.audit.segment_budget_usage == (
        (1, 1, 0),
        (1, 1, 0),
        (1, 1, 0),
    )
    assert certificate.audit.evaluated_link_object_domain_count == 3
    assert certificate.audit.evaluated_self_pair_domain_count == 0
    assert not certificate.audit.all_link_object_domains_covered
    assert not certificate.audit.all_self_pair_domains_covered
    assert not certificate.audit.checkable_collision_gates_passed
    assert any(
        row.startswith(
            "SEGMENT_SHARED_BUDGET_EXHAUSTED_BEFORE_LINK_OBJECT"
        )
        for row in certificate.audit.blockers
    )


def test_one_potential_link_object_contact_propagates_unresolved(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    touching = np.array(
        fixture[2][0].nonpad_forbidden_surface.triangles_link_m,
        copy=True,
    )
    fixture[3] = HashBoundObjectSurface(
        object_id="touching_object_fixture",
        source_asset_sha256=hashlib.sha256(
            b"touching object fixture"
        ).hexdigest(),
        geometry_sha256=triangle_surface_geometry_sha256(touching),
        ray_closure_object_geometry_sha256=(
            fixture[3].ray_closure_object_geometry_sha256
        ),
        triangles_object_m=touching,
    )

    certificate = _certify(tuple(fixture))

    assert certificate.state is (
        FullHandClosureCollisionState.NOT_CERTIFIABLE
    )
    assert not certificate.audit.checkable_collision_gates_passed
    assert any(
        row.startswith("LINK_OBJECT_PATH_UNRESOLVED:0:link_a:")
        for row in certificate.audit.blockers
    )


def test_path_direction_or_continuity_cannot_be_rebound(
    tmp_path: Path,
) -> None:
    fixture = list(_fixture(tmp_path))
    segments = list(fixture[6])
    segments[1] = replace(
        segments[1],
        direction=(0.0, -1.0, 0.0),
    )
    fixture[6] = tuple(segments)

    with pytest.raises(
        FullHandCollisionError,
        match=(
            "leaves the registered joint domain|"
            "does not start at the previous endpoint|"
            "sequential closure endpoint differs"
        ),
    ):
        _certify(tuple(fixture))


def test_hash_bound_surface_is_a_deep_immutable_snapshot() -> None:
    triangles = _source_triangles()
    expected = np.array(triangles, copy=True)
    surface = HashBoundLinkSurface(
        link_name="link_a",
        source_asset_sha256="a" * 64,
        geometry_sha256=triangle_surface_geometry_sha256(triangles),
        triangles_link_m=triangles,
    )

    triangles[:] = 99.0

    assert np.array_equal(surface.triangles_link_m, expected)
    with pytest.raises(ValueError):
        surface.triangles_link_m.setflags(write=True)


def test_contact_range_policy_consumes_every_bound_but_stays_blocked(
    tmp_path: Path,
) -> None:
    certificate = _certify_policy(_policy_fixture(tmp_path))

    assert certificate.state is (
        FullHandClosureCollisionState.NOT_CERTIFIABLE
    )
    assert certificate.audit.method_id == CONTACT_RANGE_POLICY_METHOD_ID
    assert certificate.audit.claim_limitations == (
        CONTACT_RANGE_POLICY_CLAIM_LIMITATIONS
    )
    assert certificate.audit.policy_contact_ranges_consumed
    assert not (
        certificate.audit.display_approximation_used_as_formal_evidence
    )
    assert certificate.audit.checkable_collision_gates_passed
    assert certificate.audit.all_link_object_domains_covered
    assert certificate.audit.all_self_pair_domains_covered
    assert certificate.audit.evaluated_link_object_domain_count == 3
    assert certificate.audit.certified_free_link_object_domain_count == 3
    assert certificate.audit.evaluated_self_pair_domain_count == 3
    assert certificate.audit.certified_free_self_pair_domain_count == 3
    assert certificate.audit.subdivision_intervals_used == 6
    assert certificate.audit.subdivision_intervals_remaining == 0
    assert certificate.audit.link_support_bindings == (
        ("link_a", 0),
        ("link_b", 1),
        ("link_c", 2),
    )
    assert all(
        row.motion_relation == "INDEPENDENT_SUPPORT_PHASE_PRODUCT"
        for row in certificate.self_pair_certificates
    )
    assert all(
        row.collision_domain == TERMINAL_DUAL_SURFACE_COLLISION_DOMAIN
        for row in certificate.link_object_certificates
    )
    assert (
        "V9_REPRESENTATIVE_PHASE_IS_ROOT_BRACKET_MIDPOINT_"
        "NOT_EXACT_CONTACT_ENDPOINT"
        not in certificate.audit.blockers
    )
    assert (
        "ARM_ENVIRONMENT_APPROACH_CLOSURE_LIFT_COLLISION_UNAVAILABLE"
        in certificate.audit.blockers
    )
    for pad_name, link_name in (
        ("pad_a", "link_a"),
        ("pad_b", "link_b"),
        ("pad_c", "link_c"),
    ):
        assert (
            f"{PAD_SURFACE_BLOCKER_PREFIX}:{pad_name}:{link_name}"
            in certificate.audit.blockers
        )


def test_contact_range_policy_collision_ignores_display_only_value(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    contact_sets = list(policy.possible_first_contact_sets)
    root = contact_sets[0].possible_earliest_roots[0]
    implicit = root.certificate.implicit_root
    changed_display = implicit.isolating_interval.lower + 0.25 * (
        implicit.isolating_interval.upper
        - implicit.isolating_interval.lower
    )
    changed_root = replace(
        root,
        certificate=replace(
            root.certificate,
            implicit_root=replace(
                implicit,
                display_approximation=changed_display,
            ),
        ),
    )
    contact_sets[0] = PossibleFirstContactSet.from_certified_roots(
        (changed_root,)
    )
    changed_policy = replace(
        policy,
        possible_first_contact_sets=tuple(contact_sets),
    )

    assert changed_policy.policy_sha256 == policy.policy_sha256
    reference = _certify_policy((fixture, policy, policy_audit))
    changed = _certify_policy((fixture, changed_policy, policy_audit))

    assert changed.audit.policy_sha256 == reference.audit.policy_sha256
    assert changed.audit.v9_audit_and_policy_sha256 == (
        reference.audit.v9_audit_and_policy_sha256
    )
    assert changed.audit.blockers == reference.audit.blockers
    assert changed.audit.checkable_collision_gates_passed == (
        reference.audit.checkable_collision_gates_passed
    )


def test_contact_range_policy_model_binding_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    drifted_policy = replace(policy, model_contract_sha256="f" * 64)

    with pytest.raises(
        FullHandCollisionError,
        match="differs from its V9 model/path binding",
    ):
        _certify_policy((fixture, drifted_policy, policy_audit))


def test_contact_range_policy_shared_budget_cannot_claim_full_coverage(
    tmp_path: Path,
) -> None:
    certificate = _certify_policy(
        _policy_fixture(tmp_path),
        maximum_subdivision_intervals=1,
    )

    assert certificate.state is (
        FullHandClosureCollisionState.NOT_CERTIFIABLE
    )
    assert certificate.audit.evaluated_link_object_domain_count == 1
    assert certificate.audit.evaluated_self_pair_domain_count == 0
    assert not certificate.audit.all_link_object_domains_covered
    assert not certificate.audit.all_self_pair_domains_covered
    assert not certificate.audit.checkable_collision_gates_passed
    assert any(
        row.startswith(
            "POLICY_SHARED_BUDGET_EXHAUSTED_BEFORE_LINK_OBJECT"
        )
        for row in certificate.audit.blockers
    )


def test_contact_range_screening_stops_after_first_unresolved_without_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    original = (
        full_collision_module
        .certify_moving_link_surface_separated_from_static_surface
    )
    called_links: list[str] = []

    def first_child_unresolved(**kwargs):
        child = original(**kwargs)
        called_links.append(str(kwargs["link_name"]))
        if len(called_links) != 1:
            return child
        changed_audit = replace(
            child.audit,
            certified_free_leaf_interval_count=0,
            entire_phase_covered=False,
            unresolved_reason="TEST_FORCED_UNRESOLVED",
        )
        return replace(
            child,
            state=ContinuousCollisionState.UNRESOLVED,
            certified_free_leaf_intervals=(),
            unresolved_interval=child.searched_phase,
            audit=changed_audit,
        )

    monkeypatch.setattr(
        full_collision_module,
        "certify_moving_link_surface_separated_from_static_surface",
        first_child_unresolved,
    )
    certificate = certify_full_hand_contact_range_policy_closure(
        backend=fixture[0],
        link_surfaces=fixture[1],
        terminal_forbidden_surfaces=fixture[2],
        object_surface=fixture[3],
        self_pair_inventory=fixture[4],
        sequential_closure_policy=policy,
        v9_audit=policy_audit,
        maximum_subdivision_intervals=6,
        stop_after_first_unresolved_domain=True,
    )

    assert called_links == ["link_a"]
    assert certificate.state is FullHandClosureCollisionState.NOT_CERTIFIABLE
    assert certificate.audit.evaluated_link_object_domain_count == 1
    assert certificate.audit.evaluated_self_pair_domain_count == 0
    assert certificate.audit.subdivision_intervals_remaining == 5
    assert not certificate.audit.all_link_object_domains_covered
    assert not certificate.audit.all_self_pair_domains_covered
    assert not certificate.audit.checkable_collision_gates_passed
    assert (
        "POLICY_SCREENING_SHORT_CIRCUIT_AFTER_LINK_OBJECT:link_a"
        in certificate.audit.blockers
    )
    assert any(
        row.startswith(
            "POLICY_SCREENING_SHORT_CIRCUIT_UNVISITED_SELF_PAIR:"
        )
        for row in certificate.audit.blockers
    )


def test_contact_range_reuses_only_matching_prepared_object_surface(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    prepared = prepare_static_triangle_surface(
        fixture[3].triangles_object_m,
        expected_geometry_sha256=fixture[3].geometry_sha256,
    )
    reference = _certify_policy((fixture, policy, policy_audit))
    reused = certify_full_hand_contact_range_policy_closure(
        backend=fixture[0],
        link_surfaces=fixture[1],
        terminal_forbidden_surfaces=fixture[2],
        object_surface=fixture[3],
        self_pair_inventory=fixture[4],
        sequential_closure_policy=policy,
        v9_audit=policy_audit,
        maximum_subdivision_intervals=6,
        prepared_static_object_surface=prepared,
    )

    assert reused == reference
    different = prepare_static_triangle_surface(
        fixture[3].triangles_object_m + np.asarray((1.0, 0.0, 0.0))
    )
    with pytest.raises(FullHandCollisionError, match="differs from bound"):
        certify_full_hand_contact_range_policy_closure(
            backend=fixture[0],
            link_surfaces=fixture[1],
            terminal_forbidden_surfaces=fixture[2],
            object_surface=fixture[3],
            self_pair_inventory=fixture[4],
            sequential_closure_policy=policy,
            v9_audit=policy_audit,
            maximum_subdivision_intervals=6,
            prepared_static_object_surface=different,
        )


def test_contact_range_policy_wrench_consumes_every_root_but_stays_blocked(
    tmp_path: Path,
) -> None:
    _, collision_certificate, certificate = _evaluate_policy_wrench(
        _policy_fixture(tmp_path)
    )

    assert certificate.state is ContactRangePolicyWrenchState.NOT_CERTIFIABLE
    assert certificate.audit.method_id == CONTACT_RANGE_POLICY_WRENCH_METHOD_ID
    assert certificate.audit.claim_limitations == (
        CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS
    )
    assert certificate.audit.blockers == (
        CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS
    )
    assert certificate.audit.possible_root_counts == (1, 1, 1)
    assert certificate.audit.total_possible_root_count == 3
    assert certificate.audit.root_domain_rule == (
        CONTACT_RANGE_POLICY_WRENCH_ROOT_DOMAIN_RULE
    )
    assert certificate.audit.cartesian_product_count == 1
    assert certificate.audit.cartesian_product_rule == (
        CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE
    )
    assert certificate.audit.policy_contact_root_domains_consumed
    assert certificate.audit.complete_cartesian_product_bound
    assert not (
        certificate.audit.display_approximation_used_as_formal_evidence
    )
    assert not (
        certificate.audit.finite_contact_geometry_sampling_used_as_formal_evidence
    )
    assert certificate.audit.exact_candidate_wrench_invocation_count == 0
    assert not certificate.audit.contact_range_margin_computed
    assert certificate.audit.interval_contact_jacobian_certificate_present
    assert certificate.audit.joint_position_domain_rule == (
        CONTACT_RANGE_POLICY_WRENCH_JOINT_DOMAIN_RULE
    )
    assert certificate.audit.independent_joint_names == (
        "joint_a",
        "joint_b",
        "joint_c",
    )
    assert len(certificate.audit.final_joint_position_intervals) == 3
    for pad_domain in certificate.audit.pad_domains:
        for root_domain in pad_domain.roots:
            assert root_domain.object_face_index == 0
            assert root_domain.path_local_free_side_normal_object == (
                1.0,
                0.0,
                0.0,
            )
            interval_jacobian = root_domain.interval_geometric_jacobian
            assert interval_jacobian.point_object_m == (
                root_domain.position_object_m
            )
            assert interval_jacobian.joint_position_intervals == (
                certificate.audit.final_joint_position_intervals
            )
            assert len(interval_jacobian.elements) == 6
            assert all(
                len(row) == 3 for row in interval_jacobian.elements
            )
    assert not (
        certificate.audit.parametric_wrench_lower_bound_certificate_present
    )
    assert certificate.audit.parametric_wrench_lower_bound is None
    assert certificate.audit.margin_search_invocation_count == 1
    assert certificate.audit.interval_policy_margin_certificate.state is (
        IntervalPolicyMarginState.NOT_CERTIFIABLE
    )
    assert certificate.audit.interval_policy_margin_certificate.reason == (
        "MIDPOINT_MARGIN_PROPOSAL_INFEASIBLE"
    )
    assert not certificate.audit.formal_selection_allowed
    assert certificate.task_margins is None
    assert certificate.hard_bound_minimum_task_margin is None
    assert certificate.peak_normal_force_n is None
    assert certificate.joint_torque_utilization is None
    assert len(certificate.audit.policy_collision_binding_sha256) == 64
    assert collision_certificate.audit.checkable_collision_gates_passed


def test_contact_range_policy_wrench_rejects_missing_object_face(
    tmp_path: Path,
) -> None:
    fixture, policy, _ = _policy_fixture(tmp_path)
    root = policy.possible_first_contact_sets[0].possible_earliest_roots[0]
    invalid_root = replace(root, object_face_index=1)

    with pytest.raises(
        TaskWrenchEvaluationError,
        match="object face index is outside object mesh",
    ):
        _policy_wrench_evaluator()._root_wrench_domain(
            invalid_root,
            hand_model=fixture[0].hand_model,
            interval_backend=fixture[0],
            final_joint_position_intervals=(
                IntervalBounds(0.1, 0.1),
                IntervalBounds(0.2, 0.2),
                IntervalBounds(0.3, 0.3),
            ),
            object_from_hand=np.eye(4),
        )


def _balanced_policy_wrench_success(
    policy_fixture,
    scenario_parameters_unit: np.ndarray,
):
    fixture, policy, policy_audit = policy_fixture
    evaluator = TaskWrenchEvaluator(
        object_model=_task_object_model(),
        characteristic_radius_m=0.05,
        friction_coefficient_interval=(0.8, 0.8),
        uncertainty_claim_scope=(
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        gravity_direction_object=(0.0, 0.0, -1.0),
        task_frame_rotation_object=np.eye(3),
        gravity_acceleration_m_s2=0.01,
        lift_acceleration_m_s2=0.0,
        maximum_inner_approximation_relative_error=0.1,
        cone_edge_multiplier=1,
        solver_options=_POLICY_WRENCH_LP_OPTIONS,
    )
    collision_certificate = _certify_policy(policy_fixture)
    base_certificate = evaluator.evaluate_contact_range_policy(
        policy,
        scenario_parameters_unit,
        v9_audit=policy_audit,
        hand_model=fixture[0].hand_model,
        policy_collision_certificate=collision_certificate,
    )
    assert base_certificate.state is ContactRangePolicyWrenchState.NOT_CERTIFIABLE
    center = np.asarray((100.0, 0.25, 0.25), dtype=np.float64)
    radius = 0.05
    radial = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-0.5, np.sqrt(3.0) / 2.0, 0.0),
            (-0.5, -np.sqrt(3.0) / 2.0, 0.0),
        )
    )
    balanced_domains = []
    for contact_index, (domain, direction) in enumerate(
        zip(base_certificate.audit.pad_domains, radial)
    ):
        source_root = domain.roots[0]
        point = center + radius * direction
        position = tuple(
            IntervalBounds(float(value - 1.0e-6), float(value + 1.0e-6))
            for value in point
        )
        balanced_root = replace(
            source_root,
            formal_root_sha256=hashlib.sha256(
                f"balanced-main-domain:{contact_index}".encode("utf-8")
            ).hexdigest(),
            position_object_m=position,
            path_local_free_side_normal_object=tuple(
                float(value) for value in direction
            ),
            interval_geometric_jacobian=replace(
                source_root.interval_geometric_jacobian,
                point_object_m=position,
            ),
        )
        balanced_domains.append(
            ContactRangePadWrenchDomain(
                pad_name=domain.pad_name,
                roots=(balanced_root,),
            )
        )
    binding = base_certificate.audit.task_wrench_contract_sha256

    margin_certificate = evaluator._certify_contact_range_policy_margin(
        pad_domains=tuple(balanced_domains),
        hand_model=fixture[0].hand_model,
        task_wrench_contract_sha256=binding,
    )

    assert margin_certificate.state is (
        IntervalPolicyMarginState.CERTIFIED_POSITIVE_LOWER_BOUND
    )
    assert margin_certificate.certified_margin_lower_bound is not None
    assert margin_certificate.certified_margin_lower_bound > 0.0
    assert margin_certificate.evaluation_binding_sha256 == binding
    assert margin_certificate.final_policy_wrench_certificate is not None
    margin = margin_certificate.certified_margin_lower_bound
    final = margin_certificate.final_policy_wrench_certificate
    assert margin is not None and final is not None
    peak_force = max(
        bounds.upper
        for record in final.load_certificates
        for bounds in (
            record.balance_certificate.pad_normal_force_intervals or ()
        )
    )
    assert final.maximum_joint_torque_utilization_upper is not None
    success_audit = replace(
        base_certificate.audit,
        pad_domains=tuple(balanced_domains),
        contact_range_margin_computed=True,
        parametric_wrench_lower_bound_certificate_present=True,
        parametric_wrench_lower_bound=margin,
        interval_policy_margin_certificate=margin_certificate,
        blockers=CONTACT_RANGE_POLICY_WRENCH_REMAINING_BLOCKERS,
        claim_limitations=(
            CONTACT_RANGE_POLICY_WRENCH_PARAMETRIC_CLAIM_LIMITATIONS
        ),
    )
    success_certificate = ContactRangePolicyWrenchCertificate(
        state=(
            ContactRangePolicyWrenchState.PARAMETRIC_WRENCH_CERTIFIED_NONFRICTION_UNCALIBRATED
        ),
        audit=success_audit,
        task_margins=tuple(
            margin for _index in range(success_audit.scenario_count)
        ),
        hard_bound_minimum_task_margin=margin,
        peak_normal_force_n=peak_force,
        joint_torque_utilization=(
            final.maximum_joint_torque_utilization_upper
        ),
    )

    assert success_certificate.hard_bound_minimum_task_margin == margin
    assert success_certificate.audit.blockers == (
        CONTACT_RANGE_POLICY_WRENCH_REMAINING_BLOCKERS
    )
    assert not success_certificate.audit.formal_selection_allowed
    return evaluator, collision_certificate, success_certificate


def test_main_evaluator_margin_path_certifies_independent_balanced_domains(
    tmp_path: Path,
) -> None:
    _evaluator, _collision, success = _balanced_policy_wrench_success(
        _policy_fixture(tmp_path),
        np.asarray(((0.0,), (0.5,), (1.0,)), dtype=np.float64),
    )
    assert success.hard_bound_minimum_task_margin is not None
    assert success.hard_bound_minimum_task_margin > 0.0


def test_policy_ranker_calls_range_collision_and_wrench_once_and_rejects_old_fixture(
    tmp_path: Path,
) -> None:
    policy_fixture = _ranking_policy_fixture(tmp_path)
    fixture, policy, policy_audit = policy_fixture
    generation = _policy_generation_result(policy, policy_audit)
    collision_certificate = _certify_policy(policy_fixture)
    evaluator = _policy_wrench_evaluator()
    collision_calls: list[str] = []

    def exact_collision_must_not_run(_accepted):
        raise AssertionError("exact candidate collision path must not run")

    def policy_collision(accepted):
        collision_calls.append(accepted.v9_parameter_key_hex)
        return collision_certificate

    result = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=(
            _POLICY_RANK_GENERATION_SHA256
        ),
        expected_model_contract_sha256=policy_audit.model_contract_sha256,
        wrench_evaluator=evaluator,
        hand_model=fixture[0].hand_model,
        collision_certifier=exact_collision_must_not_run,
        policy_collision_certifier=policy_collision,
    ).evaluate(generation)

    assert collision_calls == [
        generation.accepted_policies[0].v9_parameter_key_hex
    ]
    assert not result.candidate_records
    assert len(result.contact_range_policy_records) == 1
    record = result.contact_range_policy_records[0]
    assert record.sequential_closure_policy is policy
    assert record.policy_sha256 == policy.policy_sha256
    assert record.collision_invocation_count == 1
    assert record.wrench_invocation_count == 1
    assert record.collision_state == "NOT_CERTIFIABLE"
    assert record.wrench_state == "NOT_CERTIFIABLE"
    assert record.state is CandidateEvaluationState.UNRESOLVED_WRENCH
    assert record.diagnostic_metrics is None
    assert not result.diagnostic_ranked_policy_keys
    assert result.selected_contact_range_policy is None
    assert result.selected_candidate is None


def test_policy_ranker_orders_balanced_range_proof_but_cannot_select_formally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_fixture = _ranking_policy_fixture(tmp_path)
    fixture, policy, policy_audit = policy_fixture
    scenarios = deterministic_sobol(
        dimension=SCENARIO_DIMENSION,
        count=SCENARIO_COUNT,
        seed=SCENARIO_SOBOL_SEED,
    )
    evaluator, source_collision, success_wrench = (
        _balanced_policy_wrench_success(policy_fixture, scenarios)
    )
    assert success_wrench.audit.scenario_design_sha256 == (
        SCENARIO_DESIGN_SHA256
    )
    generation = _policy_generation_result(policy, policy_audit)
    complete_collision = CompleteContactRangeTrajectoryCollisionCertificate(
        method_id=COMPLETE_POLICY_CLEARANCE_METHOD_ID,
        claim_scope=COMPLETE_CLEARANCE_SCOPE,
        source_policy_collision_certificate=source_collision,
        policy_sha256=policy.policy_sha256,
        v9_policy_evidence_sha256=(
            source_collision.audit.v9_audit_and_policy_sha256
        ),
        model_contract_sha256=policy_audit.model_contract_sha256,
        trajectory_clearance_lower_bound_m=0.001,
    )
    collision_calls: list[str] = []
    wrench_calls: list[str] = []

    def exact_collision_must_not_run(_accepted):
        raise AssertionError("exact candidate collision path must not run")

    def policy_collision(accepted):
        collision_calls.append(accepted.v9_parameter_key_hex)
        return complete_collision

    def policy_wrench(
        accepted_policy,
        scenario_parameters_unit,
        *,
        v9_audit,
        hand_model,
        policy_collision_certificate,
    ):
        assert accepted_policy is policy
        assert v9_audit is policy_audit
        assert hand_model is fixture[0].hand_model
        assert policy_collision_certificate is source_collision
        digest = hashlib.sha256(
            np.asarray(
                scenario_parameters_unit,
                dtype=">f8",
            ).tobytes(order="C")
        ).hexdigest()
        assert digest == SCENARIO_DESIGN_SHA256
        assert not scenario_parameters_unit.flags.writeable
        wrench_calls.append(policy.policy_sha256)
        return success_wrench

    monkeypatch.setattr(
        evaluator,
        "evaluate_contact_range_policy",
        policy_wrench,
    )
    result = PostGenerationRankOnlyPipeline(
        expected_generation_contract_sha256=(
            _POLICY_RANK_GENERATION_SHA256
        ),
        expected_model_contract_sha256=policy_audit.model_contract_sha256,
        wrench_evaluator=evaluator,
        hand_model=fixture[0].hand_model,
        collision_certifier=exact_collision_must_not_run,
        policy_collision_certifier=policy_collision,
    ).evaluate(generation)

    key = generation.accepted_policies[0].v9_parameter_key_hex
    assert collision_calls == [key]
    assert wrench_calls == [policy.policy_sha256]
    assert len(result.contact_range_policy_records) == 1
    record = result.contact_range_policy_records[0]
    assert record.state is CandidateEvaluationState.UNCERTAINTY_SCOPE_INCOMPLETE
    assert record.diagnostic_rank == 1
    assert record.diagnostic_metrics is not None
    assert record.trajectory_clearance_lower_bound_m == 0.001
    assert result.diagnostic_ranked_policy_keys == (key,)
    assert not result.formal_ranked_policy_keys
    assert result.selected_contact_range_policy is None
    assert result.selected_candidate is None
    assert "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS" in (
        result.selection_blockers
    )


def test_contact_range_policy_wrench_rejects_margin_task_binding_drift(
    tmp_path: Path,
) -> None:
    certificate = _evaluate_policy_wrench(_policy_fixture(tmp_path))[2]
    drifted_margin = replace(
        certificate.audit.interval_policy_margin_certificate,
        evaluation_binding_sha256="f" * 64,
    )

    with pytest.raises(
        TaskWrenchEvaluationError,
        match="audit binding is malformed",
    ):
        replace(
            certificate.audit,
            interval_policy_margin_certificate=drifted_margin,
        )


def test_contact_range_policy_wrench_applies_free_side_sign(
    tmp_path: Path,
) -> None:
    fixture, policy, _ = _policy_fixture(tmp_path)
    root = policy.possible_first_contact_sets[0].possible_earliest_roots[0]
    reversed_root = replace(
        root,
        certificate=replace(
            root.certificate,
            implicit_root=replace(
                root.certificate.implicit_root,
                value_at_lower=IntervalBounds(-0.2, -0.1),
                value_at_upper=IntervalBounds(0.1, 0.2),
                derivative=IntervalBounds(1.0, 2.0),
            ),
            object_source_winding_free_side_sign=-1,
        ),
    )
    domain = _policy_wrench_evaluator()._root_wrench_domain(
        reversed_root,
        hand_model=fixture[0].hand_model,
        interval_backend=fixture[0],
        final_joint_position_intervals=(
            IntervalBounds(0.1, 0.1),
            IntervalBounds(0.2, 0.2),
            IntervalBounds(0.3, 0.3),
        ),
        object_from_hand=np.eye(4),
    )

    assert domain.path_local_free_side_normal_object == (-1.0, -0.0, -0.0)


def test_contact_range_policy_wrench_ignores_display_only_value(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    contact_sets = list(policy.possible_first_contact_sets)
    root = contact_sets[0].possible_earliest_roots[0]
    implicit = root.certificate.implicit_root
    changed_display = implicit.isolating_interval.lower + 0.75 * (
        implicit.isolating_interval.upper
        - implicit.isolating_interval.lower
    )
    changed_root = replace(
        root,
        certificate=replace(
            root.certificate,
            implicit_root=replace(
                implicit,
                display_approximation=changed_display,
            ),
        ),
    )
    contact_sets[0] = PossibleFirstContactSet.from_certified_roots(
        (changed_root,)
    )
    changed_policy = replace(
        policy,
        possible_first_contact_sets=tuple(contact_sets),
    )

    assert changed_policy.policy_sha256 == policy.policy_sha256
    reference = _evaluate_policy_wrench((fixture, policy, policy_audit))[2]
    changed = _evaluate_policy_wrench(
        (fixture, changed_policy, policy_audit)
    )[2]

    assert changed.audit.audit_sha256 == reference.audit.audit_sha256
    assert changed.audit.pad_domains == reference.audit.pad_domains
    assert changed.audit.blockers == reference.audit.blockers


def test_contact_range_policy_wrench_does_not_call_exact_point_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    collision_certificate = _certify_policy(
        (fixture, policy, policy_audit)
    )
    evaluator = _policy_wrench_evaluator()

    def _unexpected_exact_point_solver(*args, **kwargs):
        del args, kwargs
        raise AssertionError("exact-point wrench solver must not be called")

    monkeypatch.setattr(
        evaluator,
        "evaluate_task_wrench",
        _unexpected_exact_point_solver,
    )
    certificate = evaluator.evaluate_contact_range_policy(
        policy,
        np.asarray(((0.25,), (0.75,)), dtype=np.float64),
        v9_audit=policy_audit,
        hand_model=fixture[0].hand_model,
        policy_collision_certificate=collision_certificate,
    )

    assert certificate.state is ContactRangePolicyWrenchState.NOT_CERTIFIABLE
    assert certificate.audit.exact_candidate_wrench_invocation_count == 0


def test_contact_range_policy_wrench_model_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, policy, policy_audit = _policy_fixture(tmp_path)
    collision_certificate = _certify_policy(
        (fixture, policy, policy_audit)
    )
    drifted_policy = replace(policy, model_contract_sha256="f" * 64)

    with pytest.raises(
        TaskWrenchEvaluationError,
        match="differs from its wrench model binding",
    ):
        _policy_wrench_evaluator().evaluate_contact_range_policy(
            drifted_policy,
            np.asarray(((0.5,),), dtype=np.float64),
            v9_audit=policy_audit,
            hand_model=fixture[0].hand_model,
            policy_collision_certificate=collision_certificate,
        )
