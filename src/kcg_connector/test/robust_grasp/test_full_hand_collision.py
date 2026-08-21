"""Fail-closed full-hand sequential collision aggregation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

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
    PAD_SURFACE_BLOCKER_PREFIX,
    ContactRangePolicyCollisionCertificate,
    FullHandClosureCollisionState,
    FullHandCollisionError,
    HashBoundLinkSurface,
    HashBoundObjectSurface,
    SequentialClosureSegment,
    TerminalForbiddenSurface,
    certify_full_hand_contact_range_policy_closure,
    certify_full_hand_sequential_closure,
    triangle_surface_geometry_sha256,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
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
from kcg_connector.grasp.robust.object_model import file_sha256
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
    PadClosureAudit,
    RayClosureAudit,
    RayClosureEvaluation,
    _canonical_json,
    _float64_array_hex,
    _hand_model_manifest,
)


def _backend() -> DirectedIntervalKinematics:
    joints = {}
    pads = {}
    finger_joints = {}
    for index, name in enumerate(("a", "b", "c")):
        joint_name = f"joint_{name}"
        link_name = f"link_{name}"
        finger_name = f"finger_{name}"
        pad_name = f"pad_{name}"
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type="prismatic",
            parent_link="hand_base",
            child_link=link_name,
            origin_xyz_m=(0.0, 10.0 * index, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            limit=JointLimit(0.0, 1.0),
        )
        pads[pad_name] = PadGeometry(
            name=pad_name,
            finger_name=finger_name,
            link_name=link_name,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (1.0, 1.0, 1.0)),
        )
        finger_joints[finger_name] = (joint_name,)
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
    candidate = GraspCandidate.from_matrix(
        object_from_hand=np.eye(4),
        independent_joint_positions_rad=phases,
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
            "parameter_layout": list(PARAMETER_LAYOUT_PREFIX),
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
        parameter_layout=PARAMETER_LAYOUT_PREFIX,
        pad_order=pad_names,
        full_verified_pad_mesh_used=True,
        pad_face_subset_input_allowed=False,
        independent_actuation_supports=supports,
        closure_parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        closure_suffix_dominance_argument=(
            CLOSURE_SUFFIX_DOMINANCE_ARGUMENT
        ),
        preshape_joint_names=(),
        closure_open_joint_positions_rad=(0.0, 0.0, 0.0),
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
) -> tuple[SequentialClosureSegment, ...]:
    return (
        SequentialClosureSegment(
            segment_index=0,
            pad_name="pad_a",
            active_link_name="link_a",
            q_start=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            phase=IntervalBounds(0.0, 0.1),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
        SequentialClosureSegment(
            segment_index=1,
            pad_name="pad_b",
            active_link_name="link_b",
            q_start=(0.1, 0.0, 0.0),
            direction=(0.0, 1.0, 0.0),
            phase=IntervalBounds(0.0, 0.2),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
        SequentialClosureSegment(
            segment_index=2,
            pad_name="pad_c",
            active_link_name="link_c",
            q_start=(0.1, 0.2, 0.0),
            direction=(0.0, 0.0, 1.0),
            phase=IntervalBounds(0.0, 0.3),
            maximum_subdivision_intervals=(
                maximum_subdivision_intervals
            ),
        ),
    )


def _fixture(tmp_path: Path, *, budget: int = 6):
    backend = _backend()
    terminal_rows = tuple(
        _terminal_input(tmp_path, link_name=f"link_{name}")
        for name in ("a", "b", "c")
    )
    links = tuple(row[0] for row in terminal_rows)
    terminals = tuple(row[1] for row in terminal_rows)
    object_triangles = np.asarray(
        (
            (
                (100.0, 0.0, 0.0),
                (100.0, 1.0, 0.0),
                (100.0, 0.0, 1.0),
            ),
        ),
        dtype=np.float64,
    )
    object_surface = HashBoundObjectSurface(
        object_id="object_fixture",
        source_asset_sha256=hashlib.sha256(b"object fixture").hexdigest(),
        geometry_sha256=triangle_surface_geometry_sha256(
            object_triangles
        ),
        ray_closure_object_geometry_sha256=hashlib.sha256(
            b"Ray object fixture"
        ).hexdigest(),
        triangles_object_m=object_triangles,
    )
    inventory = build_self_collision_pair_inventory(
        link_names=("link_a", "link_b", "link_c"),
        srdf_assertions=(
            DisabledCollisionAssertion("link_a", "link_b", "Adjacent"),
            DisabledCollisionAssertion("link_b", "link_c", "Never"),
        ),
    )
    segments = _segments(budget)
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
            IntervalBounds(float(ordinal), float(ordinal) + 0.01),
            IntervalBounds(-0.01, 0.01),
            IntervalBounds(-0.01, 0.01),
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
        object_face_index=ordinal,
        semantic_classification=(
            "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
        ),
        certificate=certificate,
    )
    return PossibleFirstContactSet.from_certified_roots((root,))


def _policy_fixture(tmp_path: Path):
    fixture = _fixture(tmp_path)
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
        initial_independent_joint_positions_rad=(0.0, 0.0, 0.0),
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
