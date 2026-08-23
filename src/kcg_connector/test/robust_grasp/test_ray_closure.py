from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import math
from threading import local
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from kcg_connector.grasp.robust.hand_contract import (
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
    VerifiedFileReference,
    VerifiedPad,
    load_carts_hand_contract,
)
from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.object_model import (
    AssetProvenance,
    CARTS_VISUAL_SUBTREE_NPZ,
    ObjectGraspModel,
    TriangleMesh,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLAIM_LIMITATIONS,
    CLOSURE_FOCUS_METHOD,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CLOSURE_SUFFIX_DOMINANCE_ARGUMENT,
    FEATURE_ROOT_POLICY,
    INTERNAL_FORCE_ROLE,
    METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    POSSIBLE_EARLIEST_ORDERING_POLICY,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    RAY_EVALUATION_POLICY,
    TRAJECTORY_CLEARANCE_ROLE,
    PreRegisteredTaskFrame,
    RayClosureError,
    RayClosureSurfaceModel,
    WholePathPadSphereScreen,
    _Budget,
    _DirectionalWitnessSegmentBounds,
    _FK_ERROR,
    _GeometryExecutionContext,
    _IntervalGeometry,
    _PadCounters,
    _ParentPairInheritance,
    _PairIntervalClassification,
    _PadSearchState,
    _PointTriangleDistanceBvh,
    _closest_points_on_triangle,
    _closest_points_on_triangle_pairs,
    _exact_dyadic_plane_key,
    _exact_dyadic_plane_key_fraction_reference,
    _prepare_pad,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    BATCH_POINT_MOTION_METHOD_ID,
    DISPLAY_APPROXIMATION_ROLE,
    IntervalBounds,
    IntervalPointMotionBatch,
    IntervalRootState,
)


_EXTERNAL = "external_surface"
_FORBIDDEN = "forbidden_surface"
_PARAMETERS = np.asarray((0.0, 0.5, 0.5, 0.5))
_REPOSITORY = Path(__file__).resolve().parents[4]
_HAND_CONTRACT_PATH = (
    _REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
)


def _mesh_reference(name: str, points: np.ndarray, faces: np.ndarray) -> VerifiedFileReference:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    return VerifiedFileReference(
        repository_relative_path=f"synthetic/{name}.npz",
        absolute_path=Path(f"/synthetic/{name}.npz"),
        sha256=digest.hexdigest(),
        byte_count=int(points.nbytes + faces.nbytes),
    )


def _pad(name: str, finger: str, link: str) -> VerifiedPad:
    points = np.asarray(
        (
            (-0.04, -0.04, 0.0),
            (0.04, -0.04, 0.0),
            (0.04, 0.04, 0.0),
            (-0.04, 0.04, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int64)
    return VerifiedPad(
        name=name,
        finger_name=finger,
        link_name=link,
        origin_xyz_m=(0.0, 0.0, 0.0),
        origin_rpy_rad=(0.0, 0.0, 0.0),
        mesh=_mesh_reference(name, points, faces),
        coordinate_frame=link,
        unit="m",
        normal_force_capacity_n=1.0,
        points_local_m=points,
        faces=faces,
    )


def _hand_and_pads() -> tuple[ThreeFingerHandModel, tuple[VerifiedPad, ...]]:
    definitions = (
        (
            "finger_a",
            "finger_a_joint",
            "finger_a_link",
            "pad_a",
            (1.5, 0.0, 0.0),
            (0.0, math.pi / 2.0, 0.0),
        ),
        (
            "finger_b",
            "finger_b_joint",
            "finger_b_link",
            "pad_b",
            (0.0, 1.5, 0.0),
            (-math.pi / 2.0, 0.0, 0.0),
        ),
        (
            "finger_c",
            "finger_c_joint",
            "finger_c_link",
            "pad_c",
            (-1.5, 0.0, 0.0),
            (0.0, -math.pi / 2.0, 0.0),
        ),
    )
    joints: dict[str, JointSpec] = {}
    pad_geometry: dict[str, PadGeometry] = {}
    finger_joints: dict[str, tuple[str, ...]] = {}
    verified: list[VerifiedPad] = []
    for finger, joint_name, link, pad_name, origin, rpy in definitions:
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type="prismatic",
            parent_link="hand_base",
            child_link=link,
            origin_xyz_m=origin,
            origin_rpy_rad=rpy,
            axis=(0.0, 0.0, -1.0),
            limit=JointLimit(0.0, 2.0),
        )
        finger_joints[finger] = (joint_name,)
        pad_geometry[pad_name] = PadGeometry(
            name=pad_name,
            finger_name=finger,
            link_name=link,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (0.08, 0.08, 0.001)),
        )
        verified.append(_pad(pad_name, finger, link))
    hand = ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names=finger_joints,
        pads=pad_geometry,
    )
    return hand, tuple(verified)


def _object_model(
    vertices: np.ndarray,
    faces: np.ndarray,
    semantics: tuple[str, ...],
    *,
    forbidden: frozenset[str] = frozenset(),
) -> ObjectGraspModel:
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    digest.update("\0".join(semantics).encode("utf-8"))
    provenance = AssetProvenance(
        source_path="/synthetic/ray_closure_fixture.npz",
        source_sha256=digest.hexdigest(),
        source_class="SYNTHETIC_GEOMETRY_TEST",
        source_format=CARTS_VISUAL_SUBTREE_NPZ,
        source_unit="m",
        meters_per_source_unit=1.0,
    )
    return ObjectGraspModel(
        mesh=TriangleMesh(vertices, faces, semantics),
        provenance=provenance,
        assembly_axis=np.asarray((0.0, 0.0, 1.0)),
        assembly_axis_origin_m=np.zeros(3),
        mass_kg=1.0,
        center_of_mass_m=np.mean(vertices, axis=0),
        inertia_kg_m2=np.eye(3),
        allowed_contact_semantics=frozenset((_EXTERNAL,)),
        forbidden_contact_semantics=forbidden,
    )


def _box_model(
    *,
    forbid_positive_y: bool = False,
    forbid_positive_x: bool = False,
) -> ObjectGraspModel:
    vertices = np.asarray(
        (
            (-1.0, -0.75, -0.5),
            (1.0, -0.75, -0.5),
            (1.0, 0.75, -0.5),
            (-1.0, 0.75, -0.5),
            (-1.0, -0.75, 0.5),
            (1.0, -0.75, 0.5),
            (1.0, 0.75, 0.5),
            (-1.0, 0.75, 0.5),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int64,
    )
    semantics = [_EXTERNAL] * len(faces)
    forbidden = frozenset()
    if forbid_positive_y:
        semantics[6] = _FORBIDDEN
        semantics[7] = _FORBIDDEN
        forbidden = frozenset((_FORBIDDEN,))
    if forbid_positive_x:
        semantics[10] = _FORBIDDEN
        semantics[11] = _FORBIDDEN
        forbidden = frozenset((_FORBIDDEN,))
    return _object_model(vertices, faces, tuple(semantics), forbidden=forbidden)


def test_distance_bvh_aabb_query_contains_bruteforce_face_set() -> None:
    model = _box_model()
    bvh = _PointTriangleDistanceBvh(model)
    lower = np.asarray((-1.1, -0.1, -0.1))
    upper = np.asarray((0.2, 0.2, 0.2))
    reported = set(
        int(value)
        for value in bvh.face_indices_intersecting_aabb(lower, upper)
    )
    triangles = model.mesh.face_vertices_m
    face_lower = np.min(triangles, axis=1)
    face_upper = np.max(triangles, axis=1)
    reference = {
        index
        for index in range(len(triangles))
        if np.all(face_upper[index] >= lower)
        and np.all(face_lower[index] <= upper)
    }
    assert reference <= reported

    assert bvh.face_indices_intersecting_aabb(
        (4.0, 4.0, 4.0), (5.0, 5.0, 5.0)
    ).size == 0


def test_distance_bvh_packet_aabb_query_exactly_matches_scalar_queries() -> None:
    bvh = _PointTriangleDistanceBvh(_box_model())
    lower = np.asarray(
        (
            (-1.1, -0.1, -0.1),
            (4.0, 4.0, 4.0),
            (1.0, -0.75, -0.5),
            (-2.0, -2.0, -2.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            (0.2, 0.2, 0.2),
            (5.0, 5.0, 5.0),
            (1.0, 0.75, 0.5),
            (2.0, 2.0, 2.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    packet = bvh.face_indices_intersecting_aabbs(lower, upper)
    scalar = tuple(
        bvh.face_indices_intersecting_aabb(row_lower, row_upper)
        for row_lower, row_upper in zip(lower, upper)
    )

    assert len(packet) == len(scalar)
    for packet_row, scalar_row in zip(packet, scalar):
        assert np.array_equal(packet_row, scalar_row)
        assert not packet_row.flags.writeable

    with pytest.raises(RayClosureError, match="non-empty"):
        bvh.face_indices_intersecting_aabbs(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )


def _append_quad(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    semantics: list[str],
    corners: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> None:
    start = len(vertices)
    vertices.extend(corners)
    faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
    semantics.extend((_EXTERNAL, _EXTERNAL))


def _concave_open_model() -> ObjectGraspModel:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    semantics: list[str] = []

    # Positive and negative x entry surfaces.
    _append_quad(
        vertices,
        faces,
        semantics,
        ((1.0, -1.0, -0.5), (1.0, 1.0, -0.5), (1.0, 1.0, 0.5), (1.0, -1.0, 0.5)),
    )
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-1.0, 1.0, -0.5), (-1.0, -1.0, -0.5), (-1.0, -1.0, 0.5), (-1.0, 1.0, 0.5)),
    )
    # An internal x layer must not replace the external first hit.
    _append_quad(
        vertices,
        faces,
        semantics,
        ((0.5, -1.0, -0.5), (0.5, 1.0, -0.5), (0.5, 1.0, 0.5), (0.5, -1.0, 0.5)),
    )
    # Negative y establishes a symmetric object-centred lateral range.
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-1.0, -1.0, -0.5), (1.0, -1.0, -0.5), (1.0, -1.0, 0.5), (-1.0, -1.0, 0.5)),
    )
    # Four positive-y strips leave a real opening around the second PAD.
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-1.0, 1.0, -0.5), (-1.0, 1.0, 0.5), (-0.2, 1.0, 0.5), (-0.2, 1.0, -0.5)),
    )
    _append_quad(
        vertices,
        faces,
        semantics,
        ((0.2, 1.0, -0.5), (0.2, 1.0, 0.5), (1.0, 1.0, 0.5), (1.0, 1.0, -0.5)),
    )
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-0.2, 1.0, -0.5), (-0.2, 1.0, -0.2), (0.2, 1.0, -0.2), (0.2, 1.0, -0.5)),
    )
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-0.2, 1.0, 0.2), (-0.2, 1.0, 0.5), (0.2, 1.0, 0.5), (0.2, 1.0, 0.2)),
    )
    # This recessed, externally visible face is the first hit through the opening.
    _append_quad(
        vertices,
        faces,
        semantics,
        ((-0.19, 0.5, -0.19), (-0.19, 0.5, 0.19), (0.19, 0.5, 0.19), (0.19, 0.5, -0.19)),
    )
    return _object_model(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        tuple(semantics),
    )


def _planner(
    object_model: ObjectGraspModel,
    *,
    budget: int = 4096,
    directions: np.ndarray | None = None,
) -> tuple[RayClosureSurfaceModel, ThreeFingerHandModel, tuple[VerifiedPad, ...]]:
    hand, pads = _hand_and_pads()
    model = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        budget=budget,
        directions=directions,
    )
    return model, hand, pads


def _model_from_parts(
    *,
    object_model: ObjectGraspModel,
    hand_model: ThreeFingerHandModel,
    verified_pads: tuple[VerifiedPad, ...],
    budget: int = 4096,
    directions: np.ndarray | None = None,
    task_transverse_axis_object: tuple[float, float, float] = (
        1.0,
        0.0,
        0.0,
    ),
    interval_decimal_precision: int = 80,
    maximum_root_bisection_iterations: int = 256,
) -> RayClosureSurfaceModel:
    return RayClosureSurfaceModel(
        object_model=object_model,
        hand_model=hand_model,
        verified_pads=verified_pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=task_transverse_axis_object,
            source="SYNTHETIC_PRE_REGISTERED_TASK_FRAME",
        ),
        closing_actuation_directions_unit=(
            np.eye(3) if directions is None else directions
        ),
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        maximum_subdivision_intervals=budget,
        interval_decimal_precision=interval_decimal_precision,
        maximum_root_bisection_iterations=(
            maximum_root_bisection_iterations
        ),
    )


def _contact_by_pad(candidate: object) -> dict[str, object]:
    return {contact.pad_name: contact for contact in candidate.planned_pad_contacts}


def _display_candidate(evaluation: object) -> object:
    assert evaluation.candidate is None
    assert not evaluation.feasible
    assert evaluation.display_only_proposal is not None
    assert evaluation.sequential_closure_policy is not None
    assert evaluation.static_policy_available
    assert evaluation.candidate_is_representative_proposal
    assert not evaluation.exact_contact_endpoint_certified
    assert evaluation.audit.candidate_role == CANDIDATE_REPRESENTATIVE_ROLE
    assert evaluation.audit.failure_reason == (
        REPRESENTATIVE_PROPOSAL_FAILURE_REASON
    )
    assert evaluation.audit.candidate_exact_contact_endpoint_certified is False
    assert evaluation.audit.display_approximation_role == (
        DISPLAY_APPROXIMATION_ROLE
    )
    return evaluation.display_only_proposal.grasp_candidate


def _v9_first_pad_search(
    model: RayClosureSurfaceModel,
) -> tuple[object, _PadCounters, _Budget]:
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    counters = _PadCounters()
    budget = _Budget(model.maximum_subdivision_intervals)
    outcome = model._search_pad_first_contact_v9(
        prepared=prepared,
        q_start=q_start,
        direction=direction,
        maximum_parameter=model._maximum_path_parameter(
            q_start, direction
        ),
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
        budget=budget,
        counters=counters,
        execution=_GeometryExecutionContext(),
    )
    return outcome, counters, budget


def test_v9_private_search_preserves_complete_possible_earliest_set_without_rays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())

    def forbidden_ray_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("V9 contact acceptance must not call a ray")

    monkeypatch.setattr(model.intersector, "first_hit", forbidden_ray_call)
    outcome, counters, budget = _v9_first_pad_search(model)

    assert outcome.state is _PadSearchState.CERTIFIED_ROOT
    assert outcome.possible_first_contact_set is not None
    possible_set = outcome.possible_first_contact_set
    assert possible_set.ordering_policy == POSSIBLE_EARLIEST_ORDERING_POLICY
    assert outcome.interval_lower <= 0.25 <= outcome.interval_upper
    assert len(outcome.roots) == 6
    assert len(possible_set.all_certified_roots) == 6
    assert len(possible_set.possible_earliest_ordering) == 6
    assert {
        root.semantic_classification for root in outcome.roots
    } == {"ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"}
    assert all(
        root.certificate.phase.lower
        <= 0.25
        <= root.certificate.phase.upper
        for root in outcome.roots
    )
    assert counters.cofirst_root_count == 0
    assert counters.possible_earliest_root_count == 6
    assert counters.certified_contact_roots == 6
    assert counters.rays == 0
    assert counters.finite_chord_feature_candidates == 0
    assert counters.nonlinear_feature_roots_solved == 0
    # The ordered eight-segment search checks the preceding free segment and
    # both segments that share the exact contact boundary at phase 0.25.
    assert budget.used == 3

    first, second = outcome.roots[:2]
    assert first.certificate.implicit_root.equation_sha256 != (
        second.certificate.implicit_root.equation_sha256
    )
    assert first.certificate.phase.lower <= second.certificate.phase.upper
    assert second.certificate.phase.lower <= first.certificate.phase.upper
    assert first.binding_sha256 in possible_set.possible_earliest_ordering
    assert second.binding_sha256 in possible_set.possible_earliest_ordering
    assert "COFIRST" not in possible_set.ordering_policy
    assert len(possible_set.set_sha256) == 64
    with pytest.raises(RayClosureError, match="ordering contradicts"):
        replace(
            possible_set,
            possible_earliest_ordering=tuple(
                reversed(possible_set.possible_earliest_ordering)
            ),
        )


def test_v9_earlier_triangle_boundary_short_circuits_later_root() -> None:
    probe, _hand, _pads = _planner(_box_model())
    prepared = probe.prepared_pads[0]
    q_start = np.zeros(3)
    direction = probe.closing_directions_physical[0]
    object_from_hand = np.eye(4)
    states_early = probe._witness_states(
        prepared,
        q_start + 0.25 * direction,
        direction,
        object_from_hand,
    )
    states_late = probe._witness_states(
        prepared,
        q_start + 0.75 * direction,
        direction,
        object_from_hand,
    )
    early_point = states_early.positions_object_m[0]
    late_point = states_late.positions_object_m[0]
    spacing = min(
        np.linalg.norm(
            states_early.positions_object_m[1:] - early_point,
            axis=1,
        )
    )
    extent = spacing / 4.0
    boundary_triangle = np.vstack(
        (
            early_point + (0.0, 0.0, -extent),
            early_point + (0.0, extent, 0.0),
            early_point + (0.0, 0.0, extent),
        )
    )
    interior_triangle = np.vstack(
        (
            late_point + (0.0, -extent, -extent),
            late_point + (0.0, extent, -extent),
            late_point + (0.0, 0.0, extent),
        )
    )
    object_model = _object_model(
        np.vstack((boundary_triangle, interior_triangle)),
        np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        (_EXTERNAL, _EXTERNAL),
    )
    model, _hand, _pads = _planner(object_model)
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    counters = _PadCounters()
    budget = _Budget(model.maximum_subdivision_intervals)
    outcome = model._search_pad_first_contact_v9(
        prepared=prepared,
        q_start=q_start,
        direction=direction,
        maximum_parameter=1.0,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=0.0,
        budget=budget,
        counters=counters,
        execution=_GeometryExecutionContext(),
    )

    assert outcome.state is _PadSearchState.UNRESOLVED
    assert outcome.interval_lower <= 0.25 <= outcome.interval_upper
    assert outcome.unresolved_reason == (
        "TRIANGLE_BOUNDARY_NOT_STRICTLY_INTERIOR"
    )
    assert outcome.roots == ()
    # The earlier unresolved boundary is already sufficient to reject the
    # interval.  The later interior root must not be evaluated or allowed to
    # influence that fail-closed result.
    assert counters.certified_contact_roots == 0
    assert counters.competing_root_order_blocks >= 1
    assert counters.rays == 0


def test_nonround_box_produces_one_display_proposal_and_three_root_sets() -> None:
    model, hand, pads = _planner(_box_model())

    evaluation = model.evaluate_unit_parameters(_PARAMETERS)

    candidate = _display_candidate(evaluation)
    assert len(evaluation.possible_first_contact_sets) == 3
    assert evaluation.audit.possible_first_contact_set_sha256 == tuple(
        row.set_sha256 for row in evaluation.possible_first_contact_sets
    )
    # The full-path swept-AABB intersection places the translated pad_b start
    # at y=1.23.  Its first contact with y=3/4 is therefore 1.23-0.75=0.48.
    assert candidate.independent_joint_positions_rad == pytest.approx(
        (0.5, 0.48, 0.5), abs=2.0e-11
    )
    assert candidate.internal_normal_forces_n == (0.0, 0.0, 0.0)
    assert len(candidate.planned_pad_contacts) == 3
    assert evaluation.audit.method_id == METHOD_ID
    assert evaluation.audit.internal_force_role == INTERNAL_FORCE_ROLE
    assert evaluation.audit.trajectory_clearance_role == TRAJECTORY_CLEARANCE_ROLE
    assert evaluation.audit.closure_focus_method == CLOSURE_FOCUS_METHOD
    assert evaluation.audit.ray_evaluation_policy == RAY_EVALUATION_POLICY
    assert evaluation.audit.feature_root_policy == FEATURE_ROOT_POLICY
    assert (
        evaluation.audit.object_contact_normal_policy
        == OBJECT_CONTACT_NORMAL_POLICY
    )
    assert (
        evaluation.audit.pad_surface_normal_policy
        == PAD_SURFACE_NORMAL_POLICY
    )
    assert evaluation.audit.trajectory_clearance_m >= 0.0
    assert evaluation.audit.full_verified_pad_mesh_used
    assert not evaluation.audit.pad_face_subset_input_allowed
    assert evaluation.audit.independent_actuation_supports == (
        ("finger_a_joint",),
        ("finger_b_joint",),
        ("finger_c_joint",),
    )
    assert evaluation.audit.closure_parameter_domain_id == (
        CLOSURE_PARAMETER_DOMAIN_ID
    )
    assert evaluation.audit.closure_suffix_dominance_argument == (
        CLOSURE_SUFFIX_DOMINANCE_ARGUMENT
    )
    assert evaluation.audit.preshape_joint_names == ()
    assert evaluation.audit.closure_open_joint_positions_rad == (0.0, 0.0, 0.0)
    assert evaluation.audit.subdivision_intervals_used <= 4096
    assert set(CLAIM_LIMITATIONS) <= set(evaluation.audit.claim_limitations)
    assert all(row.verified_triangle_count == 2 for row in evaluation.audit.pad_audits)
    assert all(row.witness_count == 6 for row in evaluation.audit.pad_audits)
    assert all(row.distance_triangle_tests > 0 for row in evaluation.audit.pad_audits)
    assert all(
        row.interval_pair_evaluation_count > 0
        and row.certified_contact_root_count > 0
        and row.possible_earliest_root_count > 0
        and row.cofirst_root_count == 0
        and row.possible_first_contact_set_sha256 is not None
        and row.selected_root_phase_lower is not None
        and row.selected_root_phase_upper is not None
        and row.selected_normalized_closure_role
        == DISPLAY_APPROXIMATION_ROLE
        for row in evaluation.audit.pad_audits
    )
    assert all(
        row.first_hit_rays_cast == 0
        and row.acceptance_ray_call_count == 0
        and row.finite_chord_feature_candidates == 0
        and row.nonlinear_feature_roots_solved == 0
        and row.nonlinear_root_fk_evaluations == 0
        for row in evaluation.audit.pad_audits
    )


    transform = candidate.object_from_hand_matrix()
    links = hand.forward_kinematics(
        candidate.independent_joint_positions_rad, base_transform=transform
    )
    contacts = _contact_by_pad(candidate)
    pads_by_name = {pad.name: pad for pad in pads}
    for audit_row in evaluation.audit.pad_audits:
        assert audit_row.selected_triangle_index is not None
        assert audit_row.selected_witness_index is not None
        contact = contacts[audit_row.pad_name]
        pad = pads_by_name[audit_row.pad_name]
        triangle = pad.points_local_m[pad.faces[audit_row.selected_triangle_index]]
        local = np.asarray(contact.surface_coordinates[:3]) @ triangle
        link_transform = links[pad.link_name]
        final_witness = link_transform[:3, :3] @ local + link_transform[:3, 3]
        assert np.allclose(
            final_witness,
            contact.position_object_m,
            rtol=0.0,
            atol=3.0e-11,
        )

    assert contacts["pad_a"].position_object_m[0] == pytest.approx(1.0)
    assert contacts["pad_b"].position_object_m[1] == pytest.approx(0.75)
    assert contacts["pad_c"].position_object_m[0] == pytest.approx(-1.0)


def test_sequential_policy_contains_start_directions_and_contact_ranges_only() -> None:
    model, _hand, _pads = _planner(_box_model())

    evaluation = model.evaluate_unit_parameters(_PARAMETERS)
    policy = evaluation.sequential_closure_policy

    assert policy is not None
    q_start, _target, _rotation = model._decode(_PARAMETERS)
    assert policy.initial_independent_joint_positions_rad == tuple(q_start)
    assert policy.independent_joint_names == tuple(
        model.hand_model.independent_joint_names
    )
    assert policy.pad_order == tuple(
        prepared.verified.name for prepared in model.prepared_pads
    )
    assert policy.independent_actuation_supports == (
        model.independent_actuation_supports
    )
    assert policy.closing_directions_physical == (
        model.closing_directions_physical_tuple
    )
    assert policy.possible_first_contact_sets == (
        evaluation.possible_first_contact_sets
    )
    assert not hasattr(policy, "independent_joint_positions_rad")
    assert not hasattr(policy, "planned_pad_contacts")
    document = policy.as_dict()
    assert document["exact_final_joint_vector_present"] is False
    assert document["exact_contact_points_present"] is False
    assert document["display_approximation_used_as_formal_evidence"] is False
    assert "display_approximation_binary64_hex" not in json.dumps(
        document, sort_keys=True
    )
    assert len(policy.policy_sha256) == 64


def test_sequential_policy_identity_does_not_use_display_candidate_values() -> None:
    model, _hand, _pads = _planner(_box_model())
    evaluation = model.evaluate_unit_parameters(_PARAMETERS)
    policy = evaluation.sequential_closure_policy
    proposal = evaluation.display_only_proposal

    assert policy is not None
    assert proposal is not None
    changed_candidate = replace(
        proposal.grasp_candidate,
        independent_joint_positions_rad=tuple(
            value + 0.01
            for value in proposal.grasp_candidate.independent_joint_positions_rad
        ),
    )
    changed_evaluation = replace(
        evaluation,
        display_only_proposal=replace(
            proposal, grasp_candidate=changed_candidate
        ),
    )

    assert changed_evaluation.sequential_closure_policy is policy
    assert changed_evaluation.sequential_closure_policy.policy_sha256 == (
        policy.policy_sha256
    )


def test_sequential_policy_rejects_direction_or_pad_binding_drift() -> None:
    model, _hand, _pads = _planner(_box_model())
    evaluation = model.evaluate_unit_parameters(_PARAMETERS)
    policy = evaluation.sequential_closure_policy

    assert policy is not None
    directions = list(policy.closing_directions_physical)
    directions[0] = tuple(0.0 for _ in directions[0])
    with pytest.raises(RayClosureError, match="direction 0 differs"):
        replace(policy, closing_directions_physical=tuple(directions))
    with pytest.raises(RayClosureError, match="contact sets differ"):
        replace(
            policy,
            pad_order=(
                policy.pad_order[1],
                policy.pad_order[0],
                policy.pad_order[2],
            ),
        )


@pytest.mark.parametrize(
    "flipped_indices",
    (
        tuple(range(12)),
        (0, 3, 5, 8, 11),
    ),
)
def test_object_face_winding_changes_do_not_change_v9_display_proposal(
    flipped_indices: tuple[int, ...],
) -> None:
    object_model = _box_model()
    reference_model, _hand, _pads = _planner(object_model)
    reference = reference_model.evaluate_unit_parameters(_PARAMETERS)
    reference_candidate = _display_candidate(reference)

    faces = np.array(object_model.mesh.faces, dtype=np.int64, copy=True)
    rows = np.asarray(flipped_indices, dtype=np.int64)
    faces[rows, 1], faces[rows, 2] = (
        faces[rows, 2].copy(),
        faces[rows, 1].copy(),
    )
    flipped_mesh = TriangleMesh(
        vertices_m=object_model.mesh.vertices_m,
        faces=faces,
        face_semantics=object_model.mesh.face_semantics,
    )
    flipped_model, _flipped_hand, _flipped_pads = _planner(
        replace(object_model, mesh=flipped_mesh)
    )
    flipped = flipped_model.evaluate_unit_parameters(_PARAMETERS)
    reference_screen = reference_model.screen_unit_parameters(_PARAMETERS)
    flipped_screen = flipped_model.screen_unit_parameters(_PARAMETERS)

    flipped_candidate = _display_candidate(flipped)
    assert flipped_candidate == reference_candidate
    assert flipped.audit.possible_first_contact_set_sha256 == (
        reference.audit.possible_first_contact_set_sha256
    )
    assert flipped.audit.failure_reason == reference.audit.failure_reason
    assert flipped.audit.object_geometry_sha256 != (
        reference.audit.object_geometry_sha256
    )
    assert flipped.audit.model_contract_sha256 != (
        reference.audit.model_contract_sha256
    )
    assert tuple(
        (row.certified_free, row.certified_no_valid_contact)
        for row in flipped_screen
    ) == tuple(
        (row.certified_free, row.certified_no_valid_contact)
        for row in reference_screen
    )


def test_open_concavity_reaches_recess_but_hidden_layer_is_not_selected() -> None:
    object_model = _concave_open_model()
    model, _hand, _pads = _planner(object_model)

    evaluation = model.evaluate_unit_parameters(_PARAMETERS)

    contacts = _contact_by_pad(_display_candidate(evaluation))
    assert contacts["pad_a"].position_object_m[0] == pytest.approx(1.0)
    assert contacts["pad_b"].position_object_m[1] == pytest.approx(0.5)
    assert contacts["pad_c"].position_object_m[0] == pytest.approx(-1.0)
    assert all(
        row.first_contact_classification
        == "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
        for row in evaluation.audit.pad_audits
    )
    full_scan_tests = sum(
        row.exact_fk_interval_evaluations
        * row.witness_count
        * len(object_model.mesh.faces)
        for row in evaluation.audit.pad_audits
    )
    actual_tests = sum(
        row.distance_triangle_tests for row in evaluation.audit.pad_audits
    )
    assert 0 < actual_tests < full_scan_tests


def test_back_facing_pad_root_is_classified_instead_of_exhausting_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand, pads = _hand_and_pads()
    pad_a = pads[0]
    reversed_faces = np.asarray(
        pad_a.faces[:, (0, 2, 1)], dtype=np.int64
    )
    reversed_pad_a = replace(
        pad_a,
        mesh=_mesh_reference(
            "pad_a_reversed", pad_a.points_local_m, reversed_faces
        ),
        faces=reversed_faces,
    )
    model = RayClosureSurfaceModel(
        object_model=_box_model(),
        hand_model=hand,
        verified_pads=(reversed_pad_a, pads[1], pads[2]),
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=(1.0, 0.0, 0.0),
            source="SYNTHETIC_PRE_REGISTERED_TASK_FRAME",
        ),
        closing_actuation_directions_unit=np.eye(3),
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        maximum_subdivision_intervals=4096,
        interval_decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )

    evaluation = model.evaluate_unit_parameters(_PARAMETERS)

    assert evaluation.candidate is None
    assert evaluation.audit.failure_reason is not None
    assert "PAD_NORMAL_DOMAIN_REJECTED:pad_a" in evaluation.audit.failure_reason
    assert not evaluation.audit.subdivision_budget_exhausted

    monkeypatch.setattr(
        model,
        "_finite_chord_feature_event",
        lambda **_arguments: None,
    )
    terminal_fallback = model.evaluate_unit_parameters(_PARAMETERS)
    assert terminal_fallback.candidate is None
    assert terminal_fallback.audit.failure_reason is not None
    assert (
        "PAD_NORMAL_DOMAIN_REJECTED:pad_a"
        in terminal_fallback.audit.failure_reason
    )
    assert terminal_fallback.audit.as_dict() == evaluation.audit.as_dict()
    assert terminal_fallback.audit.pad_audits[0].first_hit_rays_cast == 0
    assert terminal_fallback.audit.pad_audits[0].acceptance_ray_call_count == 0
    assert not terminal_fallback.audit.subdivision_budget_exhausted


def test_common_rigid_transform_preserves_joint_solution_and_transforms_contacts() -> None:
    object_model = _box_model()
    reference_model, _hand, _pads = _planner(object_model)
    reference = reference_model.evaluate_unit_parameters(_PARAMETERS)
    reference_candidate = _display_candidate(reference)

    yaw = 0.43
    pitch = -0.37
    rz = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rx = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(pitch), -math.sin(pitch)),
            (0.0, math.sin(pitch), math.cos(pitch)),
        )
    )
    rotation = rz @ rx
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (1000.0, -2000.0, 500.0)
    transformed_object = object_model.transformed(transform)
    hand, pads = _hand_and_pads()
    transformed_model = RayClosureSurfaceModel(
        object_model=transformed_object,
        hand_model=hand,
        verified_pads=pads,
        task_frame=PreRegisteredTaskFrame(
            transverse_axis_object=tuple(rotation @ np.asarray((1.0, 0.0, 0.0))),
            source="SYNTHETIC_TRANSFORMED_PRE_REGISTERED_TASK_FRAME",
        ),
        closing_actuation_directions_unit=np.eye(3),
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        maximum_subdivision_intervals=4096,
        interval_decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    transformed = transformed_model.evaluate_unit_parameters(_PARAMETERS)
    reference_screen = reference_model.screen_unit_parameters(_PARAMETERS)
    transformed_screen = transformed_model.screen_unit_parameters(
        _PARAMETERS
    )

    transformed_candidate = _display_candidate(transformed)
    assert transformed_candidate.independent_joint_positions_rad == pytest.approx(
        reference_candidate.independent_joint_positions_rad, abs=3.0e-10
    )
    assert np.allclose(
        transformed_candidate.object_from_hand_matrix(),
        transform @ reference_candidate.object_from_hand_matrix(),
        rtol=0.0,
        atol=3.0e-10,
    )
    reference_contacts = _contact_by_pad(reference_candidate)
    transformed_contacts = _contact_by_pad(transformed_candidate)
    for pad_name, contact in reference_contacts.items():
        transformed_contact = transformed_contacts[pad_name]
        transformed_back = rotation.T @ (
            np.asarray(transformed_contact.position_object_m)
            - transform[:3, 3]
        )
        reference_normal = np.asarray(
            contact.path_local_free_side_normal_object
        )
        assert abs(
            float(
                reference_normal
                @ (transformed_back - np.asarray(contact.position_object_m))
            )
        ) <= 3.0e-10
        assert np.allclose(
            transformed_contact.path_local_free_side_normal_object,
            rotation
            @ np.asarray(contact.path_local_free_side_normal_object),
            rtol=0.0,
            atol=3.0e-12,
        )
    assert tuple(
        row.possible_earliest_ordering
        for row in transformed.possible_first_contact_sets
    ) != tuple(
        row.possible_earliest_ordering
        for row in reference.possible_first_contact_sets
    )
    assert all(
        row.ordering_policy == POSSIBLE_EARLIEST_ORDERING_POLICY
        for row in transformed.possible_first_contact_sets
    )
    assert tuple(
        (row.certified_free, row.certified_no_valid_contact)
        for row in transformed_screen
    ) == tuple(
        (row.certified_free, row.certified_no_valid_contact)
        for row in reference_screen
    )


def test_order_is_deterministic_and_public_interface_has_no_face_subset_input() -> None:
    model, hand, _pads = _planner(_box_model())

    first = model.evaluate_unit_parameters(_PARAMETERS, hand)
    second = model.evaluate_unit_parameters(_PARAMETERS.copy(), hand)
    audit_before_runtime_provenance_access = first.audit.as_dict()

    assert first == second
    assert RayClosureSurfaceModel.method_id == METHOD_ID
    assert model.method_id == METHOD_ID
    assert RayClosureSurfaceModel.closure_parameter_domain_id == (
        CLOSURE_PARAMETER_DOMAIN_ID
    )
    assert model.closure_parameter_domain_id == CLOSURE_PARAMETER_DOMAIN_ID
    assert "method_id" not in vars(model)
    assert "closure_parameter_domain_id" not in vars(model)
    assert first.audit.as_dict() == audit_before_runtime_provenance_access
    assert model.parameter_dimension == 4
    assert model.parameter_layout == (
        "assembly_axis_yaw_unit",
        "axial_target_unit",
        "lateral_task_x_unit",
        "lateral_task_y_unit",
    )
    parameters = inspect.signature(RayClosureSurfaceModel).parameters
    assert tuple(parameters) == (
        "object_model",
        "hand_model",
        "verified_pads",
        "task_frame",
        "closing_actuation_directions_unit",
        "object_contact_normal_policy",
        "pad_surface_normal_policy",
        "maximum_subdivision_intervals",
        "interval_decimal_precision",
        "maximum_root_bisection_iterations",
    )
    assert tuple(
        inspect.signature(
            RayClosureSurfaceModel.evaluate_unit_parameters
        ).parameters
    ) == ("self", "parameters_unit", "hand_model")
    assert model.contract["full_verified_pad_mesh_used"] is True
    assert model.contract["pad_face_subset_input_allowed"] is False
    assert model.contract["closure_parameter_domain_id"] == (
        CLOSURE_PARAMETER_DOMAIN_ID
    )
    assert model.contract["closure_focus_method"] == CLOSURE_FOCUS_METHOD
    assert model.contract["ray_evaluation_policy"] == RAY_EVALUATION_POLICY
    assert model.contract["feature_root_policy"] == FEATURE_ROOT_POLICY
    assert (
        model.contract["object_contact_normal_policy"]
        == OBJECT_CONTACT_NORMAL_POLICY
    )
    assert (
        model.contract["pad_surface_normal_policy"]
        == PAD_SURFACE_NORMAL_POLICY
    )
    assert model.model_binding_complete is True
    assert model.model_binding_status == MODEL_BINDING_COMPLETE_STATUS
    assert first.audit.model_binding_complete is True
    assert first.audit.model_binding_status == MODEL_BINDING_COMPLETE_STATUS
    assert first.audit.object_geometry_sha256 == model.object_geometry_sha256
    assert first.audit.model_contract_sha256 == model.model_contract_sha256
    assert len(first.audit.object_geometry_sha256) == 64
    assert len(first.audit.model_contract_sha256) == 64
    int(first.audit.object_geometry_sha256, 16)
    int(first.audit.model_contract_sha256, 16)
    assert first.audit.pad_geometry_sha256 == tuple(
        pad.mesh.sha256 for pad in model.verified_pads
    )
    assert first.audit.pad_runtime_geometry_sha256 == (
        model.pad_runtime_geometry_sha256
    )
    assert first.audit.pad_link_names == tuple(
        pad.link_name for pad in model.verified_pads
    )
    assert first.audit.closing_directions_physical == tuple(
        tuple(float(value) for value in row)
        for row in model.closing_directions_physical
    )
    assert model.contract["model_contract_sha256"] == (
        model.model_contract_sha256
    )
    audit_document = first.audit.as_dict()
    assert audit_document["model_contract_digest_method_id"] == (
        MODEL_CONTRACT_DIGEST_METHOD_ID
    )
    assert audit_document["object_geometry_sha256"] == (
        model.object_geometry_sha256
    )
    assert audit_document["pad_geometry_sha256"] == list(
        model.pad_geometry_sha256
    )
    assert audit_document["pad_link_names"] == list(model.pad_link_names)
    assert audit_document["closing_directions_physical"] == [
        list(row) for row in model.closing_directions_physical_tuple
    ]
    manifest = audit_document["model_contract_manifest"]
    assert manifest["schema"] == MODEL_CONTRACT_DIGEST_METHOD_ID
    assert set(manifest["object"]) == {
        "geometry_sha256",
        "assembly_axis",
        "assembly_axis_origin_m",
    }
    assert "object_id" not in model.model_contract_canonical_json
    assert model.object_model.provenance.source_path not in (
        model.model_contract_canonical_json
    )

    with pytest.raises(RayClosureError, match="contradicts its manifest"):
        replace(first.audit, model_contract_sha256="0" * 64)
    with pytest.raises(RayClosureError, match="contradicts audit evidence"):
        replace(first.audit, object_geometry_sha256="0" * 64)
    tampered_directions = [
        list(row) for row in first.audit.closing_directions_physical
    ]
    tampered_directions[0][0] *= -1.0
    with pytest.raises(RayClosureError, match="contradicts audit evidence"):
        replace(
            first.audit,
            closing_directions_physical=tuple(
                tuple(row) for row in tampered_directions
            ),
        )
    with pytest.raises(RayClosureError, match="partial model evidence"):
        replace(first.audit, model_binding_complete=False)


def test_contact_face_mask_is_cached_once_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getter = ObjectGraspModel.contact_face_mask.fget
    assert original_getter is not None
    access_count = 0

    def counted_getter(instance: ObjectGraspModel) -> np.ndarray:
        nonlocal access_count
        access_count += 1
        return original_getter(instance)

    monkeypatch.setattr(
        ObjectGraspModel,
        "contact_face_mask",
        property(counted_getter),
    )
    object_model = _box_model()
    model, hand, _pads = _planner(object_model)
    construction_access_count = access_count

    assert construction_access_count == 2
    assert np.array_equal(
        model._contact_face_mask,
        original_getter(object_model),
    )
    assert model._contact_face_mask.flags.writeable is False
    with pytest.raises(ValueError):
        model._contact_face_mask[0] = not model._contact_face_mask[0]

    model.evaluate_unit_parameters(_PARAMETERS, hand)
    assert access_count == construction_access_count


def test_model_contract_digest_binds_every_declared_model_input() -> None:
    object_model = _box_model()
    hand, pads = _hand_and_pads()
    reference = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
    )
    repeated = _model_from_parts(
        object_model=_box_model(),
        hand_model=hand,
        verified_pads=tuple(reversed(pads)),
    )

    changed_points = np.array(pads[0].points_local_m, copy=True)
    changed_points[0, 0] -= 0.001
    changed_pad = replace(
        pads[0],
        points_local_m=changed_points,
        mesh=_mesh_reference(
            "pad_a_geometry_variant",
            changed_points,
            pads[0].faces,
        ),
    )
    pad_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=(changed_pad, pads[1], pads[2]),
    )
    object_variant = _model_from_parts(
        object_model=_box_model(forbid_positive_y=True),
        hand_model=hand,
        verified_pads=pads,
    )
    direction_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        directions=np.diag((-1.0, 1.0, 1.0)),
    )

    changed_joint_name = hand.joint_order[0]
    changed_joints = dict(hand.joints)
    changed_joint = changed_joints[changed_joint_name]
    changed_joints[changed_joint_name] = replace(
        changed_joint,
        origin_xyz_m=(
            changed_joint.origin_xyz_m[0] + 0.01,
            changed_joint.origin_xyz_m[1],
            changed_joint.origin_xyz_m[2],
        ),
    )
    changed_hand = ThreeFingerHandModel(
        base_link=hand.base_link,
        joints=changed_joints,
        joint_order=hand.joint_order,
        finger_joint_names={
            name: chain.joint_names for name, chain in hand.fingers.items()
        },
        pads=dict(hand.pads),
    )
    hand_variant = _model_from_parts(
        object_model=object_model,
        hand_model=changed_hand,
        verified_pads=pads,
    )
    task_basis_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        task_transverse_axis_object=(0.0, 1.0, 0.0),
    )
    precision_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        interval_decimal_precision=96,
    )
    root_budget_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        maximum_root_bisection_iterations=384,
    )
    subdivision_budget_variant = _model_from_parts(
        object_model=object_model,
        hand_model=hand,
        verified_pads=pads,
        budget=2048,
    )

    assert repeated.model_contract_sha256 == reference.model_contract_sha256
    variant_digests = {
        pad_variant.model_contract_sha256,
        object_variant.model_contract_sha256,
        direction_variant.model_contract_sha256,
        hand_variant.model_contract_sha256,
        task_basis_variant.model_contract_sha256,
        precision_variant.model_contract_sha256,
        root_budget_variant.model_contract_sha256,
        subdivision_budget_variant.model_contract_sha256,
    }
    assert reference.model_contract_sha256 not in variant_digests
    assert len(variant_digests) == 8
    assert pad_variant.pad_geometry_sha256[0] != (
        reference.pad_geometry_sha256[0]
    )
    assert direction_variant.closing_directions_physical_tuple != (
        reference.closing_directions_physical_tuple
    )
    with pytest.raises(RayClosureError, match="complete hand model contract"):
        reference.evaluate_unit_parameters(_PARAMETERS, changed_hand)


def test_full_closed_area_centroid_focus_is_remesh_invariant_and_finger_balanced() -> None:
    coarse_points = np.asarray(
        (
            (-0.04, -0.04, 0.0),
            (0.04, -0.04, 0.0),
            (0.04, 0.04, 0.0),
            (-0.04, 0.04, 0.0),
        ),
        dtype=np.float64,
    )
    coarse_faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    refined_points = np.vstack((coarse_points, (0.021, 0.013, 0.0)))
    refined_faces = np.asarray(
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64
    )
    coarse_pad = replace(
        _pad("coarse_pad", "finger_a", "finger_a_link"),
        mesh=_mesh_reference("coarse_pad", coarse_points, coarse_faces),
        points_local_m=coarse_points,
        faces=coarse_faces,
    )
    refined_pad = replace(
        _pad("refined_pad", "finger_a", "finger_a_link"),
        mesh=_mesh_reference("refined_pad", refined_points, refined_faces),
        points_local_m=refined_points,
        faces=refined_faces,
    )
    coarse = _prepare_pad(coarse_pad, relevant_joint_indices=(0,))
    refined = _prepare_pad(refined_pad, relevant_joint_indices=(0,))

    assert not np.allclose(np.mean(coarse_points, axis=0), np.mean(refined_points, axis=0))
    assert np.allclose(
        coarse.surface_centroid_link_m,
        refined.surface_centroid_link_m,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    assert not coarse.surface_centroid_link_m.flags.writeable
    assert not refined.surface_centroid_link_m.flags.writeable

    model, hand, _pads = _planner(_box_model())
    q_start, _target, _rotation = model._decode(_PARAMETERS)
    focus_result = model._closure_focus_hand(q_start)
    assert focus_result is not None
    focus, _hand_extent = focus_result
    q_closed = q_start + np.sum(model.closing_directions_physical, axis=0)
    links = hand.forward_kinematics(q_closed)
    expected_centroids = []
    for prepared in model.prepared_pads:
        transform = links[prepared.verified.link_name]
        expected_centroids.append(
            transform[:3, :3] @ prepared.surface_centroid_link_m
            + transform[:3, 3]
        )
    expected_focus = np.mean(np.vstack(expected_centroids), axis=0)
    assert np.allclose(focus, expected_focus, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps)


def test_placement_domain_is_the_per_pad_swept_aabb_overlap_intersection() -> None:
    model, hand, _pads = _planner(_box_model())
    q_start = np.array(model.open_joint_template, copy=True)
    rotation = np.eye(3)

    lower, upper = model._placement_coordinate_bounds(q_start, rotation)

    assert np.all(lower <= upper)
    focus_result = model._closure_focus_hand(q_start)
    assert focus_result is not None
    focus, _hand_extent = focus_result
    expected_lower_rows = []
    expected_upper_rows = []
    for row_index, prepared in enumerate(model.prepared_pads):
        direction = model.closing_directions_physical[row_index]
        maximum_parameter = model._maximum_path_parameter(q_start, direction)
        midpoint_parameter = 0.5 * maximum_parameter
        links = hand.forward_kinematics(
            q_start + midpoint_parameter * direction
        )
        transform = links[prepared.verified.link_name]
        points = (
            prepared.verified.points_local_m @ transform[:3, :3].T
            + transform[:3, 3]
        )
        offsets = points - focus
        speed_bounds = model._local_point_speed_bounds(
            prepared,
            prepared.verified.points_local_m,
            q_start,
            direction,
            maximum_parameter,
        )
        radii = np.nextafter(
            speed_bounds * midpoint_parameter,
            math.inf,
        )
        # The production bound adds a binary64 FK term.  Reconstruct the
        # enclosing property here instead of duplicating its private gamma.
        expected_lower_rows.append(
            model.object_coordinate_lower_m
            - np.max(offsets + radii[:, None], axis=0)
        )
        expected_upper_rows.append(
            model.object_coordinate_upper_m
            - np.min(offsets - radii[:, None], axis=0)
        )
    expected_lower = np.max(np.vstack(expected_lower_rows), axis=0)
    expected_upper = np.min(np.vstack(expected_upper_rows), axis=0)
    assert np.all(lower <= expected_lower)
    assert np.all(upper >= expected_upper)
    assert not lower.flags.writeable
    assert not upper.flags.writeable


def test_swept_domain_keeps_midpath_contacts_that_closed_endpoint_excludes() -> None:
    base = _box_model()
    vertices = np.array(base.mesh.vertices_m, copy=True)
    vertices[:, 0] *= 0.1
    vertices[:, 1] *= 0.4
    vertices[:, 2] *= 0.2
    object_model = _object_model(
        vertices,
        np.array(base.mesh.faces, copy=True),
        tuple(base.mesh.face_semantics),
    )
    model, hand, _pads = _planner(object_model)
    q_start = np.array(model.open_joint_template, copy=True)
    lower, upper = model._placement_coordinate_bounds(q_start, np.eye(3))

    assert np.all(lower <= 0.0)
    assert np.all(upper >= 0.0)

    focus_result = model._closure_focus_hand(q_start)
    assert focus_result is not None
    focus, _extent = focus_result
    q_closed = q_start + np.sum(model.closing_directions_physical, axis=0)
    links = hand.forward_kinematics(q_closed)
    closed_lower_rows = []
    closed_upper_rows = []
    for prepared in model.prepared_pads:
        transform = links[prepared.verified.link_name]
        offsets = (
            prepared.verified.points_local_m @ transform[:3, :3].T
            + transform[:3, 3]
            - focus
        )
        closed_lower_rows.append(
            model.object_coordinate_lower_m - np.max(offsets, axis=0)
        )
        closed_upper_rows.append(
            model.object_coordinate_upper_m - np.min(offsets, axis=0)
        )
    closed_lower = np.max(np.vstack(closed_lower_rows), axis=0)
    closed_upper = np.min(np.vstack(closed_upper_rows), axis=0)
    assert np.any(closed_lower > closed_upper)


def test_parameter_domain_is_canonical_and_publicly_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand, _pads = _planner(_box_model())
    yaw_seam = np.array(_PARAMETERS, copy=True)
    yaw_seam[0] = 1.0

    seam = model.evaluate_unit_parameters(yaw_seam, hand)
    assert seam.candidate is None
    assert seam.audit.failure_reason is not None
    assert "canonical half-open" in seam.audit.failure_reason

    def degenerate_bounds(
        _q_start: np.ndarray,
        _rotation: np.ndarray,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        point = np.zeros(3, dtype=np.float64)
        return point, point.copy()

    monkeypatch.setattr(model, "_placement_coordinate_bounds", degenerate_bounds)
    degenerate = model.evaluate_unit_parameters(_PARAMETERS, hand)
    assert degenerate.candidate is None
    assert degenerate.audit.failure_reason is not None
    assert "zero-width placement axes" in degenerate.audit.failure_reason

    def empty_bounds(
        _q_start: np.ndarray,
        _rotation: np.ndarray,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise RayClosureError("synthetic empty intersection")

    monkeypatch.setattr(model, "_placement_coordinate_bounds", empty_bounds)
    empty = model.evaluate_unit_parameters(_PARAMETERS, hand)
    assert empty.candidate is None
    assert empty.audit.failure_reason is not None
    assert "synthetic empty intersection" in empty.audit.failure_reason


def test_shared_or_foreign_closing_support_is_a_mathematical_obstruction() -> None:
    hand, pads = _hand_and_pads()

    with pytest.raises(RayClosureError, match="shared or foreign"):
        RayClosureSurfaceModel(
            object_model=_box_model(),
            hand_model=hand,
            verified_pads=pads,
            task_frame=PreRegisteredTaskFrame(
                transverse_axis_object=(1.0, 0.0, 0.0),
                source="SYNTHETIC_PRE_REGISTERED_TASK_FRAME",
            ),
            closing_actuation_directions_unit=np.asarray(
                ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
            ),
            object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
            pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
            maximum_subdivision_intervals=4096,
            interval_decimal_precision=80,
            maximum_root_bisection_iterations=256,
        )


def test_full_open_endpoint_handles_both_closing_signs_without_joint_sampling() -> None:
    positive, _hand, _pads = _planner(_box_model(), directions=np.eye(3))
    negative, _hand, _pads = _planner(_box_model(), directions=-np.eye(3))

    positive_q = np.array(positive.open_joint_template, copy=True)
    negative_q = np.array(negative.open_joint_template, copy=True)

    assert np.array_equal(positive_q, (0.0, 0.0, 0.0))
    assert np.array_equal(negative_q, (2.0, 2.0, 2.0))
    assert all(
        positive._maximum_path_parameter(positive_q, row) == pytest.approx(1.0)
        for row in positive.closing_directions_physical
    )
    assert all(
        negative._maximum_path_parameter(negative_q, row) == pytest.approx(1.0)
        for row in negative.closing_directions_physical
    )
    positive_bounds = positive._placement_coordinate_bounds(
        positive_q, np.eye(3)
    )
    negative_bounds = negative._placement_coordinate_bounds(
        negative_q, np.eye(3)
    )
    assert np.all(positive_bounds[0] <= positive_bounds[1])
    assert np.all(negative_bounds[0] <= negative_bounds[1])


def test_v1_suffix_dominance_rejects_multi_joint_closure_support() -> None:
    contract = load_carts_hand_contract(
        _HAND_CONTRACT_PATH, repository_root=_REPOSITORY
    )
    hand = contract.build_hand_model()
    pads = tuple(sorted(contract.pads, key=lambda pad: (pad.finger_name, pad.name)))
    directions = np.zeros((3, len(hand.independent_joint_names)), dtype=np.float64)
    directions[0, hand.independent_joint_names.index("f1j1")] = 1.0
    directions[0, hand.independent_joint_names.index("f1j2")] = 1.0
    directions[1, hand.independent_joint_names.index("f2j1")] = 1.0
    directions[2, hand.independent_joint_names.index("f3j2")] = 1.0

    with pytest.raises(RayClosureError, match="exactly one independent closure joint"):
        RayClosureSurfaceModel(
            object_model=_box_model(),
            hand_model=hand,
            verified_pads=pads,
            task_frame=PreRegisteredTaskFrame(
                transverse_axis_object=(1.0, 0.0, 0.0),
                source="MULTI_SUPPORT_REJECTION_TEST",
            ),
            closing_actuation_directions_unit=directions,
            object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
            pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
            maximum_subdivision_intervals=64,
            interval_decimal_precision=80,
            maximum_root_bisection_iterations=256,
        )


def test_forbidden_first_contact_and_compute_budget_exhaustion_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_model, _hand, _pads = _planner(_box_model(forbid_positive_y=True))
    forbidden = forbidden_model.evaluate_unit_parameters(_PARAMETERS)

    assert forbidden.candidate is None
    assert forbidden.audit.failure_reason is not None
    assert "FORBIDDEN_SEMANTIC_FIRST_CONTACT:pad_b" in forbidden.audit.failure_reason
    assert not forbidden.audit.subdivision_budget_exhausted

    budget_model, _hand, _pads = _planner(_box_model(), budget=1)
    ray_calls = 0
    original_first_hit = budget_model.intersector.first_hit

    def counted_first_hit(*args: object, **kwargs: object) -> object:
        nonlocal ray_calls
        ray_calls += 1
        return original_first_hit(*args, **kwargs)

    monkeypatch.setattr(budget_model.intersector, "first_hit", counted_first_hit)
    exhausted = budget_model.evaluate_unit_parameters(_PARAMETERS)
    assert exhausted.candidate is None
    assert exhausted.audit.failure_reason == "MAXIMUM_SUBDIVISION_INTERVALS_EXHAUSTED"
    assert exhausted.audit.subdivision_budget_exhausted
    assert exhausted.audit.subdivision_intervals_used == 1
    assert exhausted.audit.maximum_subdivision_intervals == 1
    assert ray_calls == 0


def test_budget_exhaustion_never_repeats_broadphase_or_pair_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model(), budget=1)
    context = _GeometryExecutionContext()
    scalar_face_query_count = 0
    packet_face_query_count = 0
    pair_classification_count = 0
    original_scalar_face_query = (
        model.distance_bvh.face_indices_intersecting_aabb
    )
    original_packet_face_query = (
        model.distance_bvh._iter_face_pairs_intersecting_aabbs
    )

    def counted_scalar_face_query(
        *args: object, **kwargs: object
    ) -> np.ndarray:
        nonlocal scalar_face_query_count
        scalar_face_query_count += 1
        return original_scalar_face_query(*args, **kwargs)

    def counted_packet_face_query(
        *args: object, **kwargs: object
    ) -> object:
        nonlocal packet_face_query_count
        packet_face_query_count += 1
        return original_packet_face_query(*args, **kwargs)

    def forced_unresolved_batch(
        **kwargs: object,
    ) -> object:
        nonlocal pair_classification_count
        pair_classification_count += 1
        counters = kwargs["counters"]
        assert isinstance(counters, _PadCounters)
        counters.unresolved_witness_face_pairs += 1
        face_indices = np.asarray(
            kwargs["object_face_indices"], dtype=np.int64
        )
        yield _PairIntervalClassification(
                state=_PadSearchState.UNRESOLVED,
                witness_flat_index=int(kwargs["witness_flat_index"]),
                object_face_index=int(face_indices[0]),
                possible_phase_lower=float(kwargs["lower"]),
                root=None,
                reason="FORCED_UNRESOLVED_PAIR_FOR_LAZY_TRAVERSAL_TEST",
            )

    monkeypatch.setattr(
        model.distance_bvh,
        "face_indices_intersecting_aabb",
        counted_scalar_face_query,
    )
    monkeypatch.setattr(
        model.distance_bvh,
        "_iter_face_pairs_intersecting_aabbs",
        counted_packet_face_query,
    )
    monkeypatch.setattr(
        model,
        "_iter_classify_witness_face_batch_v9",
        forced_unresolved_batch,
    )

    evaluation = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        None,
        execution=context,
        _use_whole_path_sphere_screen=False,
    )

    assert evaluation.candidate is None
    assert evaluation.audit.failure_reason == (
        "MAXIMUM_SUBDIVISION_INTERVALS_EXHAUSTED"
    )
    assert evaluation.audit.subdivision_budget_exhausted
    assert evaluation.audit.subdivision_intervals_used == 1
    assert scalar_face_query_count == 0
    assert packet_face_query_count <= 1
    assert pair_classification_count <= 1


def test_display_midpoint_proposal_is_rejected_by_formal_candidate_hooks() -> None:
    model, hand, _pads = _planner(_box_model())
    first = model.evaluate_unit_parameters(_PARAMETERS)
    shifted_parameters = _PARAMETERS.copy()
    shifted_parameters[1] = 0.6
    shifted = model.evaluate_unit_parameters(shifted_parameters)
    first_proposal = _display_candidate(first)
    shifted_proposal = _display_candidate(shifted)
    assert model.candidate_from_unit_parameters(_PARAMETERS, hand) is None
    with pytest.raises(RayClosureError, match="cannot be recertified"):
        model.trajectory_clearance_m(first_proposal, hand)
    with pytest.raises(RayClosureError, match="cannot be recertified"):
        model.trajectory_clearance_m(shifted_proposal, hand)
    with pytest.raises(RayClosureError, match="requires a GraspCandidate"):
        model.trajectory_clearance_m(first.display_only_proposal, hand)

    first_contact = first_proposal.planned_pad_contacts[0]
    foreign_contact = replace(first_contact, pad_name="foreign_pad")
    foreign_candidate = replace(
        first_proposal,
        planned_pad_contacts=(
            foreign_contact,
            *first_proposal.planned_pad_contacts[1:],
        ),
    )
    with pytest.raises(RayClosureError, match="deterministic PAD order"):
        model.trajectory_clearance_m(foreign_candidate, hand)

    moved_contact = replace(
        first_contact,
        position_object_m=(
            first_contact.position_object_m[0],
            first_contact.position_object_m[1] + 0.01,
            first_contact.position_object_m[2],
        ),
    )
    tampered_candidate = replace(
        first_proposal,
        planned_pad_contacts=(
            moved_contact,
            *first_proposal.planned_pad_contacts[1:],
        ),
    )
    with pytest.raises(RayClosureError, match="cannot be recertified"):
        model.trajectory_clearance_m(tampered_candidate, hand)


def test_paired_triangle_closest_points_match_prior_scalar_kernel() -> None:
    rng = np.random.default_rng(20260823)
    points = rng.normal(size=(64, 3))
    triangles = rng.normal(size=(64, 3, 3))
    triangles[0, 2] = triangles[0, 1]
    triangles[1] = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    points[1] = np.asarray((0.25, 0.25, 0.5))

    paired = _closest_points_on_triangle_pairs(points, triangles)
    prior = np.vstack(
        [
            _closest_points_on_triangle(
                points[index : index + 1], triangles[index]
            )[0]
            for index in range(len(points))
        ]
    )

    # The paired kernel deliberately changes the floating-point reduction
    # order.  Require geometric agreement to four binary64 ulps rather than
    # bitwise identity with the scalar reference.
    np.testing.assert_allclose(
        paired,
        prior,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    with pytest.raises(RayClosureError, match="aligned finite"):
        _closest_points_on_triangle_pairs(points, triangles[:2])

    bvh = _PointTriangleDistanceBvh(_box_model())
    product_points = rng.normal(size=(17, 3))
    faces = np.arange(len(bvh.triangles), dtype=np.int64)
    product = bvh._closest_points_on_face_product(
        product_points, faces
    )
    product_reference = np.stack(
        [
            _closest_points_on_triangle(
                product_points, bvh.triangles[int(face_index)]
            )
            for face_index in faces
        ],
        axis=1,
    )
    assert np.array_equal(product, product_reference)
    with pytest.raises(RayClosureError, match="malformed"):
        bvh._closest_points_on_face_product(
            product_points, np.empty(0, dtype=np.int64)
        )


def test_batched_nearest_matches_scalar_distances_faces_ties_and_statistics() -> None:
    model, _hand, _pads = _planner(_box_model())
    rng = np.random.default_rng(20260820)
    points = np.vstack(
        (
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.5),
                    (1.0, 0.75, 0.5),
                    (3.0, -2.0, 1.0),
                )
            ),
            rng.uniform((-2.0, -1.5, -1.0), (2.0, 1.5, 1.0), size=(37, 3)),
        )
    )

    batched = model.distance_bvh.nearest_many(points)
    scalar = tuple(model.distance_bvh.nearest(point) for point in points)

    assert np.array_equal(
        batched.distances_m,
        np.asarray([row.distance_m for row in scalar]),
    )
    assert np.array_equal(
        batched.positions_m,
        np.asarray([row.position_m for row in scalar]),
    )
    assert np.array_equal(
        batched.face_indices,
        np.asarray([row.face_index for row in scalar]),
    )
    assert np.array_equal(
        batched.outward_normals,
        np.asarray([row.outward_normal for row in scalar]),
    )
    assert np.array_equal(
        batched.node_visits,
        np.asarray([row.node_visits for row in scalar]),
    )
    assert np.array_equal(
        batched.triangle_tests,
        np.asarray([row.triangle_tests for row in scalar]),
    )
    assert batched.face_indices[0] == 0
    assert not batched.distances_m.flags.writeable
    assert not batched.positions_m.flags.writeable
    with pytest.raises(RayClosureError, match="shape"):
        model.distance_bvh.nearest_many(np.empty((0, 3)))


def test_vertex_upper_bound_seed_preserves_exact_nearest_results() -> None:
    model, _hand, _pads = _planner(_box_model())
    bvh = model.distance_bvh
    rng = np.random.default_rng(20260822)
    points = np.vstack(
        (
            bvh.vertices[:8] + bvh.centre_m,
            np.mean(model.canonical_object_face_vertices_m, axis=1),
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.75, 0.5),
                    (-1.0, -0.75, -0.5),
                    (4.0, -3.0, 2.0),
                ),
                dtype=np.float64,
            ),
            rng.uniform(
                (-3.0, -2.5, -2.0),
                (3.0, 2.5, 2.0),
                size=(257, 3),
            ),
        )
    )

    seeded = bvh.nearest_many(points)
    reference = bvh.nearest_many(
        points, _use_vertex_upper_bound_seed=False
    )
    for field_name in (
        "distances_m",
        "positions_m",
        "face_indices",
        "outward_normals",
    ):
        assert np.array_equal(
            getattr(seeded, field_name), getattr(reference, field_name)
        )
    assert np.all(seeded.node_visits <= reference.node_visits)
    assert np.all(seeded.triangle_tests <= reference.triangle_tests)
    assert np.all(seeded.triangle_tests > 0)

    centered = points - bvh.centre_m
    upper, vertex_indices = bvh._vertex_surface_distance_upper_bounds(
        centered
    )
    actual_selected_vertex_distances = np.linalg.norm(
        centered - bvh.vertices[vertex_indices], axis=1
    )
    assert np.all(upper >= actual_selected_vertex_distances)
    assert not upper.flags.writeable
    assert not vertex_indices.flags.writeable

    with pytest.raises(RayClosureError, match="boolean"):
        bvh.nearest_many(
            points[:1], _use_vertex_upper_bound_seed=1  # type: ignore[arg-type]
        )


def _real_hand_structural_planner() -> tuple[
    RayClosureSurfaceModel,
    ThreeFingerHandModel,
]:
    contract = load_carts_hand_contract(
        _HAND_CONTRACT_PATH, repository_root=_REPOSITORY
    )
    hand = contract.build_hand_model()
    pads = tuple(
        sorted(contract.pads, key=lambda pad: (pad.finger_name, pad.name))
    )
    directions = np.zeros(
        (3, len(hand.independent_joint_names)), dtype=np.float64
    )
    for row_index, pad in enumerate(pads):
        independent_chain_joints = [
            joint_name
            for joint_name in hand.fingers[pad.finger_name].joint_names
            if hand.joints[joint_name].movable
            and hand.joints[joint_name].mimic is None
        ]
        assert independent_chain_joints
        selected = independent_chain_joints[-1]
        directions[row_index, hand.independent_joint_names.index(selected)] = 1.0
    return (
        RayClosureSurfaceModel(
            object_model=_box_model(),
            hand_model=hand,
            verified_pads=pads,
            task_frame=PreRegisteredTaskFrame(
                transverse_axis_object=(1.0, 0.0, 0.0),
                source="REAL_HAND_STRUCTURAL_BENCHMARK_TASK_FRAME",
            ),
            closing_actuation_directions_unit=directions,
            object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
            pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
            maximum_subdivision_intervals=64,
            interval_decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
        hand,
    )


def test_real_hand_only_samples_nonclosure_preshape_joint() -> None:
    model, hand = _real_hand_structural_planner()
    repeated_model, _repeated_hand = _real_hand_structural_planner()
    assert repeated_model.model_contract_sha256 == model.model_contract_sha256
    assert model.parameter_dimension == 5
    assert model.preshape_joint_names == ("f1j1",)
    assert model.parameter_layout[-1] == "preshape_joint_unit:f1j1"
    parameters = np.full(model.parameter_dimension, 0.5)
    q_start, _target, _rotation = model._decode(parameters)
    lower, upper = hand.joint_limit_vectors()
    by_name = {name: index for index, name in enumerate(hand.independent_joint_names)}
    assert q_start[by_name["f1j1"]] == pytest.approx(
        0.5 * (lower[by_name["f1j1"]] + upper[by_name["f1j1"]])
    )
    for name in ("f1j2", "f2j1", "f3j2"):
        assert q_start[by_name[name]] == lower[by_name[name]]

    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    motion_arguments = {
        "link_name": prepared.verified.link_name,
        "q_start": q_start,
        "direction": direction,
        "phase_lower": 0.0,
        "phase_upper": maximum_parameter,
        "base_transform": np.eye(4),
        "point_local_m": prepared.witness_points_link_m[0],
    }
    assert model.interval_kinematics.point_motion(**motion_arguments) == (
        repeated_model.interval_kinematics.point_motion(**motion_arguments)
    )


def test_real_hand_per_witness_speed_bounds_enclose_exact_jacobian_velocities() -> None:
    model, _hand = _real_hand_structural_planner()
    parameters = np.full(model.parameter_dimension, 0.5)
    q_start, _target, _rotation = model._decode(parameters)
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    bounds = model._witness_speed_bounds(
        prepared, q_start, direction, maximum_parameter
    )

    assert not bounds.flags.writeable
    assert np.ptp(bounds) > 0.0
    for phase in np.linspace(0.0, maximum_parameter, 17):
        states = model._witness_states(
            prepared,
            q_start + float(phase) * direction,
            direction,
            np.eye(4),
        )
        exact_speeds = np.linalg.norm(
            states.velocities_object_per_unit, axis=1
        )
        assert np.all(exact_speeds <= bounds)


def test_real_hand_full_pad_small_grid_uses_one_origin_jacobian_per_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand = _real_hand_structural_planner()
    original_jacobian = hand.geometric_jacobian
    calls: list[tuple[str, tuple[float, float, float]]] = []

    def counted_jacobian(
        link_name: str,
        positions: object,
        *,
        point_local_m: object = (0.0, 0.0, 0.0),
        base_transform: object = None,
    ) -> np.ndarray:
        point = tuple(float(value) for value in point_local_m)
        calls.append((link_name, point))
        return original_jacobian(
            link_name,
            positions,
            point_local_m=point_local_m,
            base_transform=base_transform,
        )

    monkeypatch.setattr(hand, "geometric_jacobian", counted_jacobian)
    lower, upper = hand.joint_limit_vectors()
    rng = np.random.default_rng(20260820)
    unit_grid = rng.uniform(0.2, 0.8, size=(2, len(lower)))
    epsilon = np.finfo(np.float64).eps
    velocity_gamma = (4096.0 * epsilon) / (1.0 - 4096.0 * epsilon)
    total_state_count = 0
    total_witness_count = 0

    for unit_row in unit_grid:
        q = lower + unit_row * (upper - lower)
        for row_index, prepared in enumerate(model.prepared_pads):
            direction = model.closing_directions_physical[row_index]
            calls_before = len(calls)
            states = model._witness_states(
                prepared, q, direction, np.eye(4)
            )
            assert len(calls) == calls_before + 1
            assert calls[-1] == (
                prepared.verified.link_name,
                (0.0, 0.0, 0.0),
            )
            assert len(states) == 3 * prepared.verified.triangle_count
            assert not states.positions_object_m.flags.writeable
            assert not states.velocities_object_per_unit.flags.writeable
            total_state_count += 1
            total_witness_count += len(states)

            sample_indices = rng.choice(len(states), size=5, replace=False)
            for flat_index in sample_indices:
                point_link = prepared.witness_points_link_m[int(flat_index)]
                point_jacobian = original_jacobian(
                    prepared.verified.link_name,
                    q,
                    point_local_m=point_link,
                    base_transform=np.eye(4),
                )
                reference_velocity = point_jacobian[:3] @ direction
                vectorised_velocity = states.velocities_object_per_unit[
                    int(flat_index)
                ]
                component_scale = (
                    np.sum(
                        np.abs(point_jacobian[:3])
                        * np.abs(direction)[None, :],
                        axis=1,
                    )
                    + np.abs(reference_velocity)
                    + np.abs(vectorised_velocity)
                )
                assert np.all(
                    np.abs(vectorised_velocity - reference_velocity)
                    <= velocity_gamma * component_scale
                )

    assert len(calls) == total_state_count == len(unit_grid) * 3
    assert total_witness_count == len(unit_grid) * sum(
        3 * prepared.verified.triangle_count
        for prepared in model.prepared_pads
    )
    assert total_witness_count > len(calls)


def test_full_nearest_shadow_preserves_candidate_and_audit_field_by_field() -> None:
    model, hand, _pads = _planner(_box_model())
    optimized_context = _GeometryExecutionContext(
        cache_enabled=True,
        verify_full_nearest=False,
    )
    reference_context = _GeometryExecutionContext(
        cache_enabled=False,
        verify_full_nearest=True,
    )

    optimized = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        hand,
        execution=optimized_context,
    )
    reference = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        hand,
        execution=reference_context,
    )

    assert _display_candidate(optimized) == _display_candidate(reference)
    assert optimized.possible_first_contact_sets == (
        reference.possible_first_contact_sets
    )
    assert optimized.audit.as_dict() == reference.audit.as_dict()
    assert reference_context.stats.reference_shadow_witness_queries > 0
    assert optimized_context.stats.reference_shadow_witness_queries == 0
    assert optimized_context.stats.interval_transform_cache_hits > 0
    assert optimized_context.stats.interval_transform_cache_misses > 0
    assert optimized_context.stats.interval_transform_cache_peak_entries > 0
    assert optimized_context.stats.interval_point_cache_hits > 0
    assert optimized_context.stats.interval_point_cache_misses > 0
    assert optimized_context.stats.interval_point_cache_peak_entries > 0
    assert reference_context.stats.interval_transform_cache_hits == 0
    assert reference_context.stats.interval_transform_cache_misses == 0
    assert reference_context.stats.interval_point_cache_hits == 0
    assert reference_context.stats.interval_point_cache_misses == 0
    assert reference_context.stats.interval_point_cache_peak_entries == 0


def test_full_pad_sphere_contains_every_source_triangle_vertex() -> None:
    model, _hand, _pads = _planner(_box_model())

    for prepared in model.prepared_pads:
        distances = np.linalg.norm(
            prepared.verified.points_local_m
            - prepared.full_pad_sphere_center_link_m,
            axis=1,
        )
        assert np.all(
            distances <= prepared.full_pad_sphere_radius_upper_m
        )
        assert prepared.full_pad_sphere_radius_upper_m > 0.0

        root = prepared.surface_sphere_nodes[
            prepared.surface_sphere_root
        ]
        assert np.array_equal(
            np.sort(root.triangle_indices),
            np.arange(prepared.verified.triangle_count),
        )
        for node in prepared.surface_sphere_nodes:
            triangles = prepared.verified.points_local_m[
                prepared.verified.faces[node.triangle_indices]
            ]
            node_distances = np.linalg.norm(
                triangles.reshape((-1, 3)) - node.center_link_m,
                axis=1,
            )
            assert np.all(node_distances <= node.radius_upper_m)
            assert np.all(
                np.abs(
                    triangles.reshape((-1, 3)) - node.center_link_m
                )
                <= node.box_half_extents_upper_m
            )
            assert 0 <= node.depth <= 3
            if node.leaf:
                assert node.left == node.right == -1
            else:
                left = prepared.surface_sphere_nodes[node.left]
                right = prepared.surface_sphere_nodes[node.right]
                assert left.depth == right.depth == node.depth + 1
                assert set(left.triangle_indices).isdisjoint(
                    set(right.triangle_indices)
                )
                assert set(left.triangle_indices) | set(
                    right.triangle_indices
                ) == set(node.triangle_indices)


def test_full_pad_aabb_tree_partitions_every_source_triangle_to_leaves() -> None:
    model, _hand, _pads = _planner(_box_model())

    for prepared in model.prepared_pads:
        nodes = prepared.surface_aabb_nodes
        root = nodes[prepared.surface_aabb_root]
        triangle_count = prepared.verified.triangle_count
        assert len(nodes) == 2 * triangle_count - 1
        assert np.array_equal(
            np.sort(root.triangle_indices),
            np.arange(triangle_count),
        )
        leaf_indices: list[int] = []
        for node in nodes:
            triangles = prepared.verified.points_local_m[
                prepared.verified.faces[node.triangle_indices]
            ]
            vertices = triangles.reshape((-1, 3))
            assert np.all(
                np.abs(vertices - node.center_link_m)
                <= node.box_half_extents_upper_m
            )
            if node.leaf:
                assert len(node.triangle_indices) == 1
                leaf_indices.append(int(node.triangle_indices[0]))
            else:
                left = nodes[node.left]
                right = nodes[node.right]
                assert left.depth == right.depth == node.depth + 1
                assert set(left.triangle_indices).isdisjoint(
                    set(right.triangle_indices)
                )
                assert set(left.triangle_indices) | set(
                    right.triangle_indices
                ) == set(node.triangle_indices)
        assert sorted(leaf_indices) == list(range(triangle_count))


def test_whole_path_dual_bvh_avoids_restarted_object_tree_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    exact_calls: list[np.ndarray] = []
    aabb_calls: list[np.ndarray] = []
    boolean_calls: list[np.ndarray] = []
    dual_pair_calls: list[np.ndarray] = []
    original_nearest = model.distance_bvh.nearest_many
    original_overlap = model.distance_bvh.face_indices_intersecting_aabbs
    original_boolean = model.distance_bvh.aabbs_have_face_overlap
    original_dual_pair = model._obb_aabb_strict_separation_mask

    def counted_nearest(points: object, **kwargs: object) -> object:
        exact_calls.append(np.asarray(points, dtype=np.float64))
        return original_nearest(points, **kwargs)

    def counted_overlap(
        lower: object, upper: object, **kwargs: object
    ) -> object:
        aabb_calls.append(np.asarray(lower, dtype=np.float64))
        return original_overlap(lower, upper, **kwargs)

    def counted_boolean(
        lower: object, upper: object, **kwargs: object
    ) -> object:
        boolean_calls.append(np.asarray(lower, dtype=np.float64))
        return original_boolean(lower, upper, **kwargs)

    def counted_dual_pair(**kwargs: object) -> np.ndarray:
        dual_pair_calls.append(
            np.asarray(kwargs["obb_centers_m"], dtype=np.float64)
        )
        return original_dual_pair(**kwargs)

    monkeypatch.setattr(
        model.distance_bvh, "nearest_many", counted_nearest
    )
    monkeypatch.setattr(
        model.distance_bvh,
        "face_indices_intersecting_aabbs",
        counted_overlap,
    )
    monkeypatch.setattr(
        model.distance_bvh,
        "aabbs_have_face_overlap",
        counted_boolean,
    )
    monkeypatch.setattr(
        model, "_obb_aabb_strict_separation_mask", counted_dual_pair
    )
    second = np.array(_PARAMETERS, copy=True)
    second[1] = 0.6
    rows = model.screen_unit_parameter_batch(
        np.vstack((_PARAMETERS, second))
    )

    assert aabb_calls == []
    assert boolean_calls == []
    assert exact_calls == []
    assert dual_pair_calls
    assert len(rows) == 2
    assert all(len(candidate_rows) == 3 for candidate_rows in rows)
    assert all(
        screen.segment_count == 8
        and screen.spatial_node_query_count >= 8
        and screen.exact_distance_query_count == 0
        and screen.nearest_surface_query_count == 0
        and 0 <= screen.maximum_spatial_depth_reached <= 1
        and math.isfinite(screen.minimum_clearance_lower_bound_m)
        for candidate_rows in rows
        for screen in candidate_rows
    )
    assert sum(len(call) for call in dual_pair_calls) == sum(
        screen.spatial_node_query_count
        for candidate_rows in rows
        for screen in candidate_rows
    )


def test_vectorized_node_speeds_and_lazy_centers_equal_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )
    original = model._local_radius_speed_bounds
    call_count = 0

    def counted(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original(**kwargs)

    monkeypatch.setattr(model, "_local_radius_speed_bounds", counted)
    coverages = model._whole_path_pad_sphere_hierarchy_coverages(
        q_start=q_start,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
    )

    assert call_count == 3
    for pad_index, (prepared, coverage) in enumerate(
        zip(model.prepared_pads, coverages)
    ):
        direction = model.closing_directions_physical[pad_index]
        maximum_parameter = model._maximum_path_parameter(
            q_start, direction
        )
        half_width = 0.5 * maximum_parameter / 8.0
        for node_index, node in enumerate(
            prepared.surface_sphere_nodes
        ):
            scalar_speed = float(
                original(
                    prepared=prepared,
                    local_radii_m=(node.maximum_vertex_radius_link_m,),
                    q_start=q_start,
                    direction=direction,
                    maximum_parameter=maximum_parameter,
                )[0]
            )
            expected_radius = float(
                np.nextafter(
                    node.radius_upper_m + scalar_speed * half_width,
                    math.inf,
                )
            )
            assert coverage.node_radius_upper_m[node_index] == (
                expected_radius
            )
            expected_box_half_extents = np.nextafter(
                node.box_half_extents_upper_m
                + scalar_speed * half_width,
                math.inf,
            )
            assert np.array_equal(
                coverage.node_box_half_extents_upper_m[node_index],
                expected_box_half_extents,
            )
            for segment_index in (0, 7):
                midpoint = (segment_index + 0.5) * (
                    maximum_parameter / 8.0
                )
                transform = model.hand_model.forward_kinematics(
                    q_start + midpoint * direction,
                    base_transform=object_from_hand,
                )[prepared.verified.link_name]
                eager_center = (
                    transform[:3, :3] @ node.center_link_m
                    + transform[:3, 3]
                )
                lazy_center = (
                    coverage.rotations_object_from_link[segment_index]
                    @ node.center_link_m
                    + coverage.translations_object_m[segment_index]
                )
                assert np.array_equal(lazy_center, eager_center)
                segment_start = segment_index * (
                    maximum_parameter / 8.0
                )
                for fraction in np.linspace(0.0, 1.0, 5):
                    phase = segment_start + float(fraction) * (
                        maximum_parameter / 8.0
                    )
                    sampled_transform = (
                        model.hand_model.forward_kinematics(
                            q_start + phase * direction,
                            base_transform=object_from_hand,
                        )[prepared.verified.link_name]
                    )
                    triangles = prepared.verified.points_local_m[
                        prepared.verified.faces[node.triangle_indices]
                    ]
                    sampled_vertices = (
                        triangles.reshape((-1, 3))
                        @ sampled_transform[:3, :3].T
                        + sampled_transform[:3, 3]
                    )
                    coordinates = (
                        sampled_vertices - lazy_center
                    ) @ coverage.rotations_object_from_link[segment_index]
                    assert np.all(
                        np.abs(coordinates)
                        <= coverage.node_box_half_extents_upper_m[
                            node_index
                        ]
                        + 1.0e-12
                    )


def test_batch_aabb_overlap_boolean_matches_complete_face_query() -> None:
    model, _hand, _pads = _planner(_box_model())
    lower = np.asarray(
        (
            (-2.0, -2.0, -2.0),
            (-1.01, -0.1, -0.1),
            (10.0, 10.0, 10.0),
            (-0.2, -0.2, -0.2),
            (0.99, 0.70, 0.49),
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            (2.0, 2.0, 2.0),
            (-0.99, 0.1, 0.1),
            (10.1, 10.1, 10.1),
            (0.2, 0.2, 0.2),
            (1.01, 0.80, 0.51),
        ),
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            len(
                model.distance_bvh.face_indices_intersecting_aabb(
                    lower_row, upper_row
                )
            )
            > 0
            for lower_row, upper_row in zip(lower, upper)
        ],
        dtype=bool,
    )

    actual = model.distance_bvh.aabbs_have_face_overlap(lower, upper)

    assert np.array_equal(actual, expected)


def test_triangle_obb_sat_matches_scalar_reference_and_keeps_touching() -> None:
    yaw = 0.37
    rotation = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    center = np.asarray((0.4, -0.2, 0.3), dtype=np.float64)
    half = np.asarray((1.0, 0.6, 0.3), dtype=np.float64)
    local_triangles = np.asarray(
        (
            ((1.4, 0.0, 0.0), (1.4, 0.2, 0.0), (1.4, 0.0, 0.2)),
            ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)),
            ((1.0, -0.2, -0.1), (1.0, 0.2, -0.1), (1.0, 0.0, 0.1)),
            ((0.8, 0.8, 0.0), (1.2, 0.8, 0.0), (1.0, 1.1, 0.0)),
        ),
        dtype=np.float64,
    )
    triangles = local_triangles @ rotation.T + center

    def scalar_separated(local_triangle: np.ndarray) -> bool:
        edges = (
            local_triangle[1] - local_triangle[0],
            local_triangle[2] - local_triangle[1],
            local_triangle[0] - local_triangle[2],
        )
        axes = [*np.eye(3)]
        axes.append(np.cross(edges[0], -edges[2]))
        axes.extend(
            np.cross(edge, axis)
            for edge in edges
            for axis in np.eye(3)
        )
        for axis in axes:
            if np.linalg.norm(axis) <= np.finfo(np.float64).tiny:
                continue
            projections = local_triangle @ axis
            radius = float(np.abs(axis) @ half)
            error = 1.0e-12 * max(
                1.0,
                abs(float(np.min(projections))),
                abs(float(np.max(projections))),
                radius,
            )
            if (
                float(np.min(projections)) > radius + error
                or float(np.max(projections)) < -radius - error
            ):
                return True
        return False

    expected = np.asarray(
        [scalar_separated(row) for row in local_triangles],
        dtype=bool,
    )
    actual = RayClosureSurfaceModel._triangle_obb_strict_separation_mask(
        triangles_object_m=triangles,
        box_center_object_m=center,
        box_axes_object=rotation,
        box_half_extents_m=half,
    )

    assert np.array_equal(actual, expected)
    assert actual[0]
    assert not actual[1]
    assert not actual[2]


def test_batched_triangle_obb_pairs_equal_repeated_single_box_sat() -> None:
    yaw_rows = np.asarray((0.11, -0.29, 0.47, 0.0), dtype=np.float64)
    rotations = np.asarray(
        [
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
            for yaw in yaw_rows
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        (
            (0.4, -0.2, 0.3),
            (-0.3, 0.1, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, -0.2),
        ),
        dtype=np.float64,
    )
    half_extents = np.asarray(
        (
            (1.0, 0.6, 0.3),
            (0.3, 0.4, 0.5),
            (0.8, 0.2, 0.4),
            (0.5, 0.5, 0.5),
        ),
        dtype=np.float64,
    )
    local_triangles = np.asarray(
        (
            ((1.4, 0.0, 0.0), (1.4, 0.2, 0.0), (1.4, 0.0, 0.2)),
            ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)),
            ((0.8, 0.2, 0.0), (1.1, 0.2, 0.0), (0.8, 0.5, 0.0)),
            ((0.5, -0.2, -0.1), (0.5, 0.2, -0.1), (0.5, 0.0, 0.1)),
        ),
        dtype=np.float64,
    )
    triangles = np.einsum(
        "nvj,nkj->nvk", local_triangles, rotations
    ) + centers[:, None, :]
    expected = np.asarray(
        [
            RayClosureSurfaceModel._triangle_obb_strict_separation_mask(
                triangles_object_m=triangles[index : index + 1],
                box_center_object_m=centers[index],
                box_axes_object=rotations[index],
                box_half_extents_m=half_extents[index],
            )[0]
            for index in range(len(triangles))
        ],
        dtype=bool,
    )
    actual = (
        RayClosureSurfaceModel._triangle_obb_pair_strict_separation_mask(
            triangles_object_m=triangles,
            box_centers_object_m=centers,
            box_axes_object=rotations,
            box_half_extents_m=half_extents,
        )
    )

    assert np.array_equal(actual, expected)
    assert not actual[-1]


def test_batched_obb_aabb_sat_matches_scalar_axes_and_keeps_touching() -> None:
    angles = np.asarray((0.0, 0.31, -0.42, 0.73), dtype=np.float64)
    rotations = np.asarray(
        [
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
            for angle in angles
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        (
            (3.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.7, 1.6, 0.0),
        ),
        dtype=np.float64,
    )
    obb_half = np.asarray(
        (
            (0.5, 0.5, 0.5),
            (0.6, 0.2, 0.4),
            (0.5, 0.5, 0.5),
            (0.7, 0.1, 0.2),
        ),
        dtype=np.float64,
    )
    lower = np.tile((-0.5, -0.5, -0.5), (4, 1)).astype(np.float64)
    upper = np.tile((0.5, 0.5, 0.5), (4, 1)).astype(np.float64)

    def scalar_separated(index: int) -> bool:
        rotation = rotations[index]
        obb_axes = tuple(rotation[:, axis] for axis in range(3))
        world_axes = tuple(np.eye(3)[axis] for axis in range(3))
        axes = [*obb_axes, *world_axes]
        axes.extend(
            np.cross(obb_axis, world_axis)
            for obb_axis in obb_axes
            for world_axis in world_axes
        )
        signs = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        obb_corners = (
            signs * obb_half[index]
        ) @ rotation.T + centers[index]
        aabb_corners = lower[index] + 0.5 * (signs + 1.0) * (
            upper[index] - lower[index]
        )
        scale = max(
            1.0,
            float(np.max(np.abs(obb_corners))),
            float(np.max(np.abs(aabb_corners))),
        )
        error = 64.0 * _FK_ERROR * scale
        for axis in axes:
            if np.linalg.norm(axis) <= np.finfo(np.float64).tiny:
                continue
            obb_projection = obb_corners @ axis
            aabb_projection = aabb_corners @ axis
            if (
                float(np.max(obb_projection))
                < float(np.min(aabb_projection)) - error
                or float(np.max(aabb_projection))
                < float(np.min(obb_projection)) - error
            ):
                return True
        return False

    expected = np.asarray(
        [scalar_separated(index) for index in range(len(centers))],
        dtype=bool,
    )
    actual = RayClosureSurfaceModel._obb_aabb_strict_separation_mask(
        obb_centers_m=centers,
        obb_axes=rotations,
        obb_half_extents_m=obb_half,
        aabb_lower_m=lower,
        aabb_upper_m=upper,
    )

    assert np.array_equal(actual, expected)
    assert actual[0]
    assert not actual[1]
    assert not actual[2]


def test_moving_triangle_sat_uses_vertex_motion_bounds_and_keeps_touching() -> None:
    fixed = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    moving = np.asarray(
        (
            fixed + (0.0, 0.0, 2.0),
            fixed,
            fixed + (0.0, 0.0, 0.2),
            fixed + (3.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    radii = np.asarray(
        (
            (0.1, 0.1, 0.1),
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.25),
            (0.2, 0.2, 0.2),
        ),
        dtype=np.float64,
    )
    actual = (
        RayClosureSurfaceModel._moving_triangle_triangle_strict_separation_mask(
            moving_triangles_midpoint_m=moving,
            moving_vertex_motion_radius_upper_m=radii,
            fixed_triangles_m=np.repeat(fixed[None, :, :], 4, axis=0),
        )
    )

    assert actual.tolist() == [True, False, False, True]

    midpoint = moving[0]
    radius = radii[0]
    edges = np.stack(
        (
            midpoint[1] - midpoint[0],
            midpoint[2] - midpoint[1],
            midpoint[0] - midpoint[2],
        )
    )
    fixed_edges = np.stack(
        (
            fixed[1] - fixed[0],
            fixed[2] - fixed[1],
            fixed[0] - fixed[2],
        )
    )
    axes = np.vstack(
        (
            np.cross(edges[0], -edges[2]),
            np.cross(fixed_edges[0], -fixed_edges[2]),
            np.cross(edges[:, None, :], fixed_edges[None, :, :]).reshape(
                (-1, 3)
            ),
        )
    )
    axis_norm = np.linalg.norm(axes, axis=1)
    midpoint_projection = midpoint @ axes.T
    enclosure_lower = np.min(
        midpoint_projection - radius[:, None] * axis_norm[None, :],
        axis=0,
    )
    enclosure_upper = np.max(
        midpoint_projection + radius[:, None] * axis_norm[None, :],
        axis=0,
    )
    for phase in np.linspace(0.0, 1.0, 65):
        displacement = np.asarray(
            (
                (0.0, 0.0, 0.1 * math.sin(math.pi * phase)),
                (0.1 * math.sin(math.pi * phase), 0.0, 0.0),
                (0.0, 0.1 * math.sin(math.pi * phase), 0.0),
            )
        )
        sampled_projection = (midpoint + displacement) @ axes.T
        assert np.all(sampled_projection >= enclosure_lower - 1.0e-14)
        assert np.all(sampled_projection <= enclosure_upper + 1.0e-14)


def test_dual_bvh_and_restarted_reference_match_candidate_rejections() -> None:
    model, _hand, _pads = _planner(_box_model())
    parameter_rows = np.vstack(
        (_PARAMETERS, np.asarray((0.0, 0.6, 0.5, 0.5)))
    )
    coverages = []
    for parameter_row in parameter_rows:
        q_start, target, rotation = model._decode(parameter_row)
        transform_result = model._object_from_hand(q_start, target, rotation)
        assert transform_result is not None
        object_from_hand, hand_extent = transform_result
        spatial_error = (
            model.intersector.distance_error_bound_m
            + model.distance_bvh.aabb_error_bound_m
            + _FK_ERROR
            * (model.intersector.characteristic_length_m + hand_extent)
        )
        coverages.extend(
            model._whole_path_pad_aabb_hierarchy_coverages(
                q_start=q_start,
                object_from_hand=object_from_hand,
                spatial_error_bound_m=spatial_error,
            )
        )

    dual = model._classify_whole_path_pad_aabb_hierarchies(coverages)
    reference = (
        model._classify_whole_path_pad_aabb_hierarchies_restarted_reference(
            coverages
        )
    )

    assert tuple(row.pad_name for row in dual) == tuple(
        row.pad_name for row in reference
    )
    assert all(
        dual_row.certified_free
        for dual_row, reference_row in zip(dual, reference)
        if reference_row.certified_free
    )
    assert sum(row.certified_free for row in dual) >= sum(
        row.certified_free for row in reference
    )


def test_full_pad_aabb_depth_first_frontier_stops_after_contact_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )
    coverages = model._whole_path_pad_aabb_hierarchy_coverages(
        q_start=q_start,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
    )

    monkeypatch.setattr(
        model,
        "_obb_aabb_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["obb_centers_m"]), dtype=bool
        ),
    )
    monkeypatch.setattr(
        model,
        "_triangle_obb_pair_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["triangles_object_m"]), dtype=bool
        ),
    )
    monkeypatch.setattr(
        model,
        "_moving_triangle_triangle_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["fixed_triangles_m"]), dtype=bool
        ),
    )
    screens = model._classify_whole_path_pad_aabb_hierarchies(
        coverages
    )

    for prepared, screen in zip(model.prepared_pads, screens):
        full_cross_product_work = (
            8
            * len(prepared.surface_aabb_nodes)
            * model.distance_bvh.node_count
        )
        assert not screen.certified_free
        assert screen.spatial_node_query_count < full_cross_product_work
        assert screen.obb_sat_triangle_test_count > 0
        assert screen.temporal_refined_leaf_pair_count > 0
        assert screen.temporal_refinement_transform_count > 0
        assert screen.maximum_temporal_refinement_depth_reached == 2


def test_bounded_narrowphase_exhaustion_remains_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )
    coverages = model._whole_path_pad_aabb_hierarchy_coverages(
        q_start=q_start,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
    )

    monkeypatch.setattr(
        model,
        "_obb_aabb_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["obb_centers_m"]), dtype=bool
        ),
    )
    monkeypatch.setattr(
        model,
        "_triangle_obb_pair_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["triangles_object_m"]), dtype=bool
        ),
    )
    monkeypatch.setattr(
        model,
        "_moving_triangle_triangle_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["fixed_triangles_m"]), dtype=bool
        ),
    )
    screens = model._classify_whole_path_pad_aabb_hierarchies(
        coverages,
        enable_moving_triangle_refinement=True,
        maximum_moving_triangle_pair_tests_per_coverage=1,
    )

    assert all(not screen.certified_free for screen in screens)
    assert all(
        screen.narrowphase_work_budget_exhausted for screen in screens
    )
    assert all(
        1 <= screen.moving_triangle_sat_pair_test_count <= 64
        for screen in screens
    )


def test_batch_screen_reuses_one_full_closed_focus_per_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    original = model._closure_focus_hand
    call_count = 0

    def counted(q_start: np.ndarray) -> tuple[np.ndarray, float] | None:
        nonlocal call_count
        call_count += 1
        return original(q_start)

    monkeypatch.setattr(model, "_closure_focus_hand", counted)
    second = np.asarray((0.0, 0.6, 0.5, 0.5), dtype=np.float64)
    screens = model.screen_unit_parameter_batch(
        np.vstack((_PARAMETERS, second))
    )

    assert len(screens) == 2
    assert call_count == 2


def test_batch_screen_production_path_uses_only_one_cheap_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    calls: list[tuple[bool, bool]] = []
    reject_in_cheap_stage = False

    monkeypatch.setattr(
        model,
        "_pad_aabb_root_global_overlap_count",
        lambda _coverage: 1,
    )

    def classified(
        coverages: object,
        *,
        enable_moving_triangle_refinement: bool = True,
        maximum_moving_triangle_pair_tests_per_coverage: int | None = None,
        enable_directional_contact_feasibility: bool = False,
    ) -> tuple[WholePathPadSphereScreen, ...]:
        del maximum_moving_triangle_pair_tests_per_coverage
        calls.append(
            (
                enable_moving_triangle_refinement,
                enable_directional_contact_feasibility,
            )
        )
        result = []
        for coverage in coverages:
            prepared = model.prepared_pads[coverage.prepared_pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=8,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=1,
                    distance_triangle_tests=1,
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=reject_in_cheap_stage,
                    narrowphase_refinement_used=(
                        enable_moving_triangle_refinement
                    ),
                    directional_contact_feasibility_used=(
                        enable_directional_contact_feasibility
                    ),
                    certified_no_valid_contact=(
                        enable_directional_contact_feasibility
                    ),
                )
            )
        return tuple(result)

    monkeypatch.setattr(
        model,
        "_classify_whole_path_pad_aabb_hierarchies",
        classified,
    )
    cheap_only = model.screen_unit_parameter_batch(_PARAMETERS[None, :])
    assert calls == [
        (False, False),
        (False, False),
        (False, False),
    ]
    assert not any(
        item.directional_contact_feasibility_used
        or item.certified_no_valid_contact
        for item in cheap_only[0]
    )
    assert not any(
        item.narrowphase_refinement_used for item in cheap_only[0]
    )
    assert all(item.root_overlap_segment_count == 1 for item in cheap_only[0])

    calls.clear()
    reject_in_cheap_stage = True
    cheap_rejected = model.screen_unit_parameter_batch(
        _PARAMETERS[None, :]
    )
    assert calls == [(False, False)]
    assert not any(
        item.directional_contact_feasibility_used
        for item in cheap_rejected[0]
    )


def test_interval_dot_upper_keeps_tangent_uncertainty() -> None:
    fixed_positive_x = np.asarray(((1.0, 0.0, 0.0),))
    fixed_negative_x = np.asarray(((-1.0, 0.0, 0.0),))
    zero = np.zeros((1, 3), dtype=np.float64)
    positive_velocity = np.asarray(((1.0, 0.0, 0.0),))

    positive = RayClosureSurfaceModel._box_dot_product_upper(
        first_lower=fixed_positive_x,
        first_upper=fixed_positive_x,
        second_lower=positive_velocity,
        second_upper=positive_velocity,
    )
    negative = RayClosureSurfaceModel._box_dot_product_upper(
        first_lower=fixed_negative_x,
        first_upper=fixed_negative_x,
        second_lower=positive_velocity,
        second_upper=positive_velocity,
    )
    tangent = RayClosureSurfaceModel._box_dot_product_upper(
        first_lower=fixed_positive_x,
        first_upper=fixed_positive_x,
        second_lower=zero,
        second_upper=zero,
    )

    assert positive[0] > 0.0
    assert negative[0] < 0.0
    assert tangent[0] > 0.0


def test_directional_impossibility_is_not_reported_collision_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )
    coverages = model._whole_path_pad_aabb_hierarchy_coverages(
        q_start=q_start,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
    )

    monkeypatch.setattr(
        model,
        "_obb_aabb_strict_separation_mask",
        lambda **kwargs: np.zeros(
            len(kwargs["obb_centers_m"]), dtype=bool
        ),
    )

    def no_pad_approach(
        coverage: object,
    ) -> _DirectionalWitnessSegmentBounds:
        prepared = model.prepared_pads[coverage.prepared_pad_index]
        witness_count = len(prepared.witness_points_link_m)
        node_count = len(prepared.surface_aabb_nodes)
        return _DirectionalWitnessSegmentBounds(
            pad_approach_possible=np.zeros(
                (8, witness_count), dtype=bool
            ),
            node_pad_approach_possible=np.zeros(
                (8, node_count), dtype=bool
            ),
            interval_witness_motion_evaluation_count=16 * witness_count,
        )

    monkeypatch.setattr(
        model,
        "_directional_witness_segment_bounds",
        no_pad_approach,
    )
    screens = model._classify_whole_path_pad_aabb_hierarchies(
        coverages,
        enable_moving_triangle_refinement=False,
        enable_directional_contact_feasibility=True,
    )

    assert all(not screen.certified_free for screen in screens)
    assert all(screen.certified_no_valid_contact for screen in screens)
    assert all(
        screen.directional_bvh_node_pair_rejected_count > 0
        for screen in screens
    )
    assert all(
        screen.moving_triangle_sat_pair_test_count == 0
        for screen in screens
    )


def test_lazy_affine_node_speed_bound_dominates_independent_scalar_formula() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, _target, _rotation = model._decode(_PARAMETERS)
    radii = np.asarray((0.0, 0.01, 0.2, 1.0), dtype=np.float64)

    for pad_index, prepared in enumerate(model.prepared_pads):
        direction = model.closing_directions_physical[pad_index]
        maximum_parameter = model._maximum_path_parameter(
            q_start, direction
        )
        actual = model._local_radius_speed_bounds(
            prepared=prepared,
            local_radii_m=radii,
            q_start=q_start,
            direction=direction,
            maximum_parameter=maximum_parameter,
        )
        endpoint = q_start + maximum_parameter * direction
        resolved_start = model.hand_model.resolve_joint_positions(q_start)
        resolved_end = model.hand_model.resolve_joint_positions(endpoint)
        resolved_velocity = model.hand_model.resolve_joint_velocities(
            direction, enforce_limits=False
        )
        ancestor_names = model._ancestor_joint_names(
            prepared.verified.link_name
        )
        reference: list[float] = []
        for radius in radii:
            speed = 0.0
            for ancestor_index, name in enumerate(ancestor_names):
                joint = model.hand_model.joints[name]
                rate = abs(float(resolved_velocity[name]))
                if joint.joint_type in ("revolute", "continuous"):
                    reach = float(radius)
                    for downstream_name in ancestor_names[
                        ancestor_index + 1 :
                    ]:
                        downstream = model.hand_model.joints[
                            downstream_name
                        ]
                        reach += float(
                            np.linalg.norm(downstream.origin_xyz_m)
                        )
                        if downstream.joint_type == "prismatic":
                            reach += max(
                                abs(float(resolved_start[downstream_name])),
                                abs(float(resolved_end[downstream_name])),
                            )
                    speed += rate * reach
                elif joint.joint_type == "prismatic":
                    speed += rate
            reference.append(
                float(
                    np.nextafter(
                        speed * (1.0 + _FK_ERROR), math.inf
                    )
                )
            )
        assert np.all(actual >= np.asarray(reference))


def test_staged_pad_screen_uses_root_score_only_for_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    score_by_pad = {0: 2, 1: 0, 2: 1}
    observed_order: list[int] = []

    monkeypatch.setattr(
        model,
        "_pad_aabb_root_global_overlap_count",
        lambda coverage: score_by_pad[coverage.prepared_pad_index],
    )

    def complete_classifier(
        coverages: object,
        **_kwargs: object,
    ) -> tuple[WholePathPadSphereScreen, ...]:
        rows = tuple(coverages)
        result = []
        for coverage in rows:
            pad_index = coverage.prepared_pad_index
            observed_order.append(pad_index)
            prepared = model.prepared_pads[pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=8,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=1,
                    distance_triangle_tests=0,
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=True,
                    spatial_node_query_count=1,
                    aabb_certified_free_node_count=1,
                )
            )
        return tuple(result)

    monkeypatch.setattr(
        model,
        "_classify_whole_path_pad_aabb_hierarchies",
        complete_classifier,
    )
    screens = model.screen_unit_parameters(_PARAMETERS)

    assert screens[1].certified_free
    assert screens[1].segment_count == 8
    assert screens[1].aabb_certified_free_node_count == 1
    assert observed_order == [1]
    assert screens[0].skipped_due_to_other_pad_free
    assert screens[2].skipped_due_to_other_pad_free
    assert screens[0].segment_count == screens[2].segment_count == 0


def test_staged_pad_screen_checks_all_pads_in_deterministic_root_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    score_by_pad = {0: 3, 1: 1, 2: 2}
    observed_order: list[int] = []

    monkeypatch.setattr(
        model,
        "_pad_aabb_root_global_overlap_count",
        lambda coverage: score_by_pad[coverage.prepared_pad_index],
    )

    def uncertain_classifier(
        coverages: object,
        **_kwargs: object,
    ) -> tuple[WholePathPadSphereScreen, ...]:
        rows = tuple(coverages)
        result: list[WholePathPadSphereScreen] = []
        for coverage in rows:
            pad_index = coverage.prepared_pad_index
            observed_order.append(pad_index)
            prepared = model.prepared_pads[pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=8,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=0,
                    distance_triangle_tests=0,
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=False,
                )
            )
        return tuple(result)

    monkeypatch.setattr(
        model,
        "_classify_whole_path_pad_aabb_hierarchies",
        uncertain_classifier,
    )
    screens = model.screen_unit_parameters(_PARAMETERS)

    assert observed_order == [1, 2, 0]
    assert all(not screen.certified_free for screen in screens)
    assert all(
        not screen.skipped_due_to_other_pad_free for screen in screens
    )


def test_whole_path_sphere_uncertainty_preserves_exact_contact_result() -> None:
    model, hand, _pads = _planner(_box_model())
    optimized = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        hand,
        execution=_GeometryExecutionContext(),
    )
    reference = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        hand,
        execution=_GeometryExecutionContext(),
        _use_whole_path_sphere_screen=False,
    )

    assert _display_candidate(optimized) == _display_candidate(reference)
    assert optimized.possible_first_contact_sets == (
        reference.possible_first_contact_sets
    )
    assert optimized.audit.failure_reason == reference.audit.failure_reason
    assert all(
        row.whole_path_sphere_screen_segment_count == 8
        and row.whole_path_sphere_screen_query_count >= 8
        and not row.whole_path_sphere_screen_certified_free
        for row in optimized.audit.pad_audits
    )
    assert all(
        row.whole_path_sphere_screen_segment_count == 0
        and row.whole_path_sphere_screen_query_count == 0
        for row in reference.audit.pad_audits
    )
    optimized_document = optimized.audit.as_dict()
    reference_document = reference.audit.as_dict()
    screen_fields = {
        "whole_path_sphere_screen_segment_count",
        "whole_path_sphere_screen_query_count",
        "whole_path_sphere_screen_bvh_node_visits",
        "whole_path_sphere_screen_triangle_tests",
        "whole_path_sphere_screen_obb_sat_certified_free_node_count",
        "whole_path_sphere_screen_obb_sat_triangle_test_count",
        "whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count",
        "whole_path_sphere_screen_moving_triangle_sat_pair_test_count",
        "whole_path_sphere_screen_temporal_refined_leaf_pair_count",
        "whole_path_sphere_screen_temporal_refinement_transform_count",
        "whole_path_sphere_screen_maximum_temporal_refinement_depth_reached",
        "whole_path_sphere_screen_narrowphase_refinement_used",
        "whole_path_sphere_screen_narrowphase_work_budget_exhausted",
        "whole_path_sphere_screen_directional_contact_feasibility_used",
        "whole_path_sphere_screen_directional_bvh_node_pair_test_count",
        "whole_path_sphere_screen_directional_bvh_node_pair_rejected_count",
        "whole_path_sphere_screen_directional_leaf_face_pair_test_count",
        "whole_path_sphere_screen_directional_leaf_face_pair_rejected_count",
        "whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count",
        "whole_path_sphere_screen_certified_no_valid_contact",
        "whole_path_sphere_screen_certified_free",
        "whole_path_sphere_screen_clearance_lower_bound_m",
    }
    for document in (optimized_document, reference_document):
        for pad_document in document["pad_audits"]:
            for field_name in screen_fields:
                pad_document.pop(field_name)
    assert optimized_document == reference_document


def test_failure_first_screen_skips_exact_earlier_fingers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    screens = tuple(
        WholePathPadSphereScreen(
            pad_name=prepared.verified.name,
            finger_name=prepared.verified.finger_name,
            segment_count=8,
            nearest_surface_query_count=8,
            distance_bvh_node_visits=17,
            distance_triangle_tests=9,
            minimum_clearance_lower_bound_m=(
                0.025 if row_index == 1 else -0.001
            ),
            certified_free=row_index == 1,
        )
        for row_index, prepared in enumerate(model.prepared_pads)
    )
    monkeypatch.setattr(
        model,
        "_classify_whole_path_pad_aabb_hierarchies",
        lambda _coverages, **_kwargs: screens,
    )

    def forbidden_exact_search(**_kwargs: object) -> object:
        raise AssertionError("exact PAD search must be skipped")

    monkeypatch.setattr(
        model, "_search_pad_first_contact_v9", forbidden_exact_search
    )
    evaluation = model.evaluate_unit_parameters(_PARAMETERS)

    assert evaluation.candidate is None
    assert evaluation.audit.failure_reason == "NO_FIRST_CONTACT_FOR_PAD:pad_b"
    assert evaluation.audit.subdivision_intervals_used == 0
    assert len(evaluation.audit.pad_audits) == 1
    row = evaluation.audit.pad_audits[0]
    assert row.pad_name == "pad_b"
    assert row.whole_path_sphere_screen_certified_free
    assert row.whole_path_sphere_screen_clearance_lower_bound_m == pytest.approx(
        0.025
    )


def test_batch_linear_cull_is_fail_closed_and_keeps_triangle_boundaries() -> None:
    model, _hand, _pads = _planner(_box_model())
    for face_index, triangle in enumerate(
        model.canonical_object_face_vertices_m
    ):
        reference = (
            model.interval_kinematics.object_triangle_affine_form_bounds(
                triangle
            )
        )
        for row_index, row in enumerate(reference):
            for coefficient_index, coefficient in enumerate(row):
                assert (
                    model._object_contact_affine_lower[
                        face_index, row_index, coefficient_index
                    ]
                    <= coefficient.lower
                )
                assert (
                    model._object_contact_affine_upper[
                        face_index, row_index, coefficient_index
                    ]
                    >= coefficient.upper
                )
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    witness_index = 0
    motion = model.interval_kinematics.point_motion(
        link_name=prepared.verified.link_name,
        q_start=q_start,
        direction=direction,
        phase_lower=0.0,
        phase_upper=maximum_parameter,
        base_transform=object_from_hand,
        point_local_m=prepared.witness_points_link_m[witness_index],
    )
    start_positions = model._witness_positions_object(
        prepared, q_start, object_from_hand
    )
    end_positions = model._witness_positions_object(
        prepared,
        q_start + maximum_parameter * direction,
        object_from_hand,
    )
    tube_lower, tube_upper = (
        model._certified_chord_tube_position_bounds_v9(
            motion=motion,
            start_position_object_m=start_positions[witness_index],
            end_position_object_m=end_positions[witness_index],
            phase_lower=0.0,
            phase_upper=maximum_parameter,
            endpoint_error_bound_m=1.0e-12,
        )
    )
    for phase in np.linspace(0.0, maximum_parameter, 33):
        sampled = model._witness_positions_object(
            prepared,
            q_start + phase * direction,
            object_from_hand,
        )[witness_index]
        assert np.all(sampled >= tube_lower)
        assert np.all(sampled <= tube_upper)
    position_lower = np.asarray(
        [row.lower for row in motion.position_object_m], dtype=np.float64
    )
    position_upper = np.asarray(
        [row.upper for row in motion.position_object_m], dtype=np.float64
    )
    face_indices = np.arange(
        len(model.canonical_object_face_vertices_m), dtype=np.int64
    )
    free_mask = model._certified_batch_free_face_mask_v9(
        position_lower=position_lower,
        position_upper=position_upper,
        face_indices=face_indices,
    )
    assert np.any(free_mask)
    chord_free_mask = model._certified_chord_batch_free_face_mask_v9(
        start_position_object_m=start_positions[witness_index],
        end_position_object_m=end_positions[witness_index],
        tube_lower_object_m=tube_lower,
        tube_upper_object_m=tube_upper,
        face_indices=face_indices,
    )
    assert np.any(chord_free_mask)
    assert np.all(chord_free_mask | ~free_mask)
    triangle_index = int(prepared.triangle_indices[witness_index])
    pad_face = np.asarray(
        prepared.verified.faces[triangle_index], dtype=np.int64
    )
    pad_triangle = np.asarray(
        prepared.verified.points_local_m[pad_face], dtype=np.float64
    )
    for face_index in face_indices[chord_free_mask]:
        row = model.interval_kinematics.certify_transverse_contact_root(
            link_name=prepared.verified.link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=0.0,
            phase_upper=maximum_parameter,
            base_transform=object_from_hand,
            witness_point_local_m=(
                prepared.witness_points_link_m[witness_index]
            ),
            pad_triangle_local_m=pad_triangle,
            object_triangle_m=(
                model.canonical_object_face_vertices_m[int(face_index)]
            ),
        )
        assert row.state is IntervalRootState.CERTIFIED_FREE

    face_index = 0
    triangle = model.canonical_object_face_vertices_m[face_index]
    centroid = np.mean(triangle, axis=0)
    vertex = np.asarray(triangle[0], dtype=np.float64)
    outside = 2.0 * triangle[0] - triangle[1]
    singleton_face = np.asarray((face_index,), dtype=np.int64)
    for boundary_point in (centroid, vertex):
        boundary_mask = model._certified_batch_free_face_mask_v9(
            position_lower=boundary_point,
            position_upper=boundary_point,
            face_indices=singleton_face,
        )
        assert not bool(boundary_mask[0])
        chord_boundary_mask = (
            model._certified_chord_batch_free_face_mask_v9(
                start_position_object_m=boundary_point,
                end_position_object_m=boundary_point,
                tube_lower_object_m=boundary_point,
                tube_upper_object_m=boundary_point,
                face_indices=singleton_face,
            )
        )
        assert not bool(chord_boundary_mask[0])
    outside_mask = model._certified_batch_free_face_mask_v9(
        position_lower=outside,
        position_upper=outside,
        face_indices=singleton_face,
    )
    assert bool(outside_mask[0])


def test_chord_affine_bounds_preserve_coordinate_correlation() -> None:
    coefficient_lower = np.zeros((1, 4, 4), dtype=np.float64)
    coefficient_upper = np.zeros((1, 4, 4), dtype=np.float64)
    correlated_outside = np.asarray((1.0, -1.0, 0.0, -1.0))
    coefficient_lower[0, 1] = correlated_outside
    coefficient_upper[0, 1] = correlated_outside
    coefficient_lower[0, 2:, 3] = 1.0
    coefficient_upper[0, 2:, 3] = 1.0
    start = np.asarray((-1.0, -1.0, 0.0))
    end = np.asarray((1.0, 1.0, 0.0))
    box_lower = np.minimum(start, end)
    box_upper = np.maximum(start, end)

    broad_lower, broad_upper = (
        RayClosureSurfaceModel._outward_affine_form_bounds(
            position_lower=box_lower,
            position_upper=box_upper,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
    )
    chord_lower, chord_upper = (
        RayClosureSurfaceModel._certified_chord_affine_form_bounds_v9(
            start_position_object_m=start,
            end_position_object_m=end,
            tube_lower_object_m=box_lower,
            tube_upper_object_m=box_upper,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
    )

    assert broad_lower[0, 1] < 0.0 < broad_upper[0, 1]
    assert chord_upper[0, 1] < 0.0


def test_pairwise_chord_face_mask_exactly_matches_scalar_rows() -> None:
    model, _hand, _pads = _planner(_box_model())
    faces = np.arange(
        len(model.canonical_object_face_vertices_m), dtype=np.int64
    )
    centroids = np.mean(
        model.canonical_object_face_vertices_m[faces], axis=1
    )
    start = np.array(centroids, copy=True)
    start[1::2] += np.asarray((0.05, 0.04, 0.03))
    end = start + np.linspace(0.0, 1.0e-4, len(faces))[:, None] * np.asarray(
        (1.0, -0.5, 0.25)
    )
    radius = np.linspace(0.0, 2.0e-6, len(faces))[:, None]
    tube_lower = np.nextafter(
        np.minimum(start, end) - radius, -math.inf
    )
    tube_upper = np.nextafter(
        np.maximum(start, end) + radius, math.inf
    )

    pairwise = model._certified_chord_pairwise_free_face_mask_v9(
        start_positions_object_m=start,
        end_positions_object_m=end,
        tube_lower_object_m=tube_lower,
        tube_upper_object_m=tube_upper,
        face_indices=faces,
    )
    scalar = np.asarray(
        [
            bool(
                model._certified_chord_batch_free_face_mask_v9(
                    start_position_object_m=start[index],
                    end_position_object_m=end[index],
                    tube_lower_object_m=tube_lower[index],
                    tube_upper_object_m=tube_upper[index],
                    face_indices=faces[index : index + 1],
                )[0]
            )
            for index in range(len(faces))
        ],
        dtype=bool,
    )

    assert np.array_equal(pairwise, scalar)
    assert np.any(pairwise)
    assert np.any(~pairwise)
    assert not pairwise.flags.writeable


def test_pairwise_exact_point_affine_bounds_match_general_boxes() -> None:
    model, _hand, _pads = _planner(_box_model())
    faces = np.arange(
        len(model.canonical_object_face_vertices_m), dtype=np.int64
    )
    points = np.mean(
        model.canonical_object_face_vertices_m[faces], axis=1
    ) + np.linspace(-1.0e-4, 1.0e-4, len(faces))[:, None]
    coefficient_lower = model._object_contact_affine_lower[faces]
    coefficient_upper = model._object_contact_affine_upper[faces]

    general_lower, general_upper = (
        model._outward_affine_form_bounds_pairwise(
            position_lower=points,
            position_upper=points,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
    )
    point_lower, point_upper = (
        model._outward_affine_form_bounds_at_points_pairwise(
            positions=points,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
    )

    assert np.array_equal(point_lower, general_lower)
    assert np.array_equal(point_upper, general_upper)


def test_packet_motion_bvh_and_pair_order_match_scalar_reference() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    possible_indices = np.arange(
        len(prepared.witness_points_link_m), dtype=np.int64
    )
    start_positions = model._witness_positions_object(
        prepared, q_start, object_from_hand
    )
    end_positions = model._witness_positions_object(
        prepared,
        q_start + maximum_parameter * direction,
        object_from_hand,
    )
    endpoint_error_bound_m = 1.0e-12

    reference_pairs: list[tuple[int, int]] = []
    reference_cache = model.interval_kinematics.new_link_transform_cache()
    reference_swept_count = 0
    for witness_index_value in possible_indices:
        witness_index = int(witness_index_value)
        motion = model.interval_kinematics.point_motion(
            link_name=prepared.verified.link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=0.0,
            phase_upper=maximum_parameter,
            base_transform=object_from_hand,
            point_local_m=prepared.witness_points_link_m[witness_index],
            transform_cache=reference_cache,
        )
        broad_lower = np.asarray(
            [row.lower for row in motion.position_object_m], dtype=np.float64
        )
        broad_upper = np.asarray(
            [row.upper for row in motion.position_object_m], dtype=np.float64
        )
        face_indices = model.distance_bvh.face_indices_intersecting_aabb(
            broad_lower, broad_upper
        )
        tube_lower, tube_upper = (
            model._certified_chord_tube_position_bounds_v9(
                motion=motion,
                start_position_object_m=start_positions[witness_index],
                end_position_object_m=end_positions[witness_index],
                phase_lower=0.0,
                phase_upper=maximum_parameter,
                endpoint_error_bound_m=endpoint_error_bound_m,
            )
        )
        query_lower = np.nextafter(
            tube_lower - model.distance_bvh.centre_m, -math.inf
        )
        query_upper = np.nextafter(
            tube_upper - model.distance_bvh.centre_m, math.inf
        )
        tube_overlap = np.all(
            model.distance_bvh.face_upper_m[face_indices] >= query_lower,
            axis=1,
        ) & np.all(
            model.distance_bvh.face_lower_m[face_indices] <= query_upper,
            axis=1,
        )
        face_indices = face_indices[tube_overlap]
        reference_swept_count += len(face_indices)
        if len(face_indices) > 0:
            free_mask = model._certified_chord_batch_free_face_mask_v9(
                start_position_object_m=start_positions[witness_index],
                end_position_object_m=end_positions[witness_index],
                tube_lower_object_m=tube_lower,
                tube_upper_object_m=tube_upper,
                face_indices=face_indices,
            )
            face_indices = face_indices[~free_mask]
        reference_pairs.extend(
            (witness_index, int(face_index))
            for face_index in face_indices
        )

    packet_counters = _PadCounters()
    packet_pairs = tuple(
        model._iter_complete_swept_face_pairs_v9(
            prepared=prepared,
            possible_witness_indices=possible_indices,
            q_start=q_start,
            direction=direction,
            lower=0.0,
            upper=maximum_parameter,
            object_from_hand=object_from_hand,
            counters=packet_counters,
            transform_cache=(
                model.interval_kinematics.new_link_transform_cache()
            ),
            apply_certified_batch_cull=True,
            witness_start_positions_object_m=start_positions,
            witness_end_positions_object_m=end_positions,
            endpoint_error_bound_m=endpoint_error_bound_m,
        )
    )

    assert packet_pairs == tuple(reference_pairs)
    assert packet_counters.swept_face_candidates == reference_swept_count
    assert packet_counters.interval_point_motion_evaluations == len(
        possible_indices
    )
    assert packet_counters.swept_face_witness_stages > 1
    assert packet_counters.swept_face_witnesses_materialized == len(
        possible_indices
    )


def test_child_interval_parent_frontier_matches_complete_query() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    midpoint = 0.5 * maximum_parameter
    possible_indices = np.arange(
        len(prepared.witness_points_link_m), dtype=np.int64
    )

    def batches(
        lower: float,
        upper: float,
        parent: Mapping[int, np.ndarray] | None,
        executor: ThreadPoolExecutor | None = None,
    ) -> tuple[dict[int, np.ndarray], _PadCounters]:
        start_positions = model._witness_positions_object(
            prepared,
            q_start + lower * direction,
            object_from_hand,
        )
        end_positions = model._witness_positions_object(
            prepared,
            q_start + upper * direction,
            object_from_hand,
        )
        counters = _PadCounters()
        rows = dict(
            model._iter_complete_swept_face_batches_v9(
                prepared=prepared,
                possible_witness_indices=possible_indices,
                q_start=q_start,
                direction=direction,
                lower=lower,
                upper=upper,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                apply_certified_batch_cull=True,
                witness_start_positions_object_m=start_positions,
                witness_end_positions_object_m=end_positions,
                endpoint_error_bound_m=1.0e-12,
                parent_face_frontier=parent,
                pair_cull_executor=executor,
            )
        )
        return rows, counters

    parent_rows, _parent_counters = batches(
        0.0, maximum_parameter, None
    )
    complete_child_rows, complete_child_counters = batches(
        0.0, midpoint, None
    )
    reused_child_rows, reused_child_counters = batches(
        0.0, midpoint, parent_rows
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_child_rows, parallel_child_counters = batches(
            0.0, midpoint, parent_rows, executor
        )

    assert set(reused_child_rows) == set(complete_child_rows)
    for witness_index, complete_faces in complete_child_rows.items():
        assert np.array_equal(
            reused_child_rows[witness_index], complete_faces
        )
        assert np.array_equal(
            parallel_child_rows[witness_index], complete_faces
        )
    assert set(parallel_child_rows) == set(complete_child_rows)
    assert reused_child_counters.swept_face_candidates <= (
        complete_child_counters.swept_face_candidates
    )
    assert parallel_child_counters == reused_child_counters


def test_staged_temporal_defer_rechecks_unmaterialized_later_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    witness_count = len(prepared.witness_points_link_m)
    later_witness = witness_count - 1
    geometry_calls = 0
    iterator_calls: list[tuple[tuple[int, ...], object]] = []

    def controlled_geometry(**_kwargs: object) -> _IntervalGeometry:
        nonlocal geometry_calls
        geometry_calls += 1
        possible = np.zeros(witness_count, dtype=bool)
        if geometry_calls == 1:
            possible[:] = True
        elif geometry_calls == 2:
            possible[later_witness] = True
        possible.setflags(write=False)
        nearest = np.zeros(witness_count, dtype=np.int64)
        nearest.setflags(write=False)
        return _IntervalGeometry(
            possible=possible,
            nearest_face_indices=nearest,
            minimum_free_margin_m=(0.0 if np.any(possible) else 1.0),
        )

    def controlled_batches(**kwargs: object) -> object:
        possible = np.asarray(
            kwargs["possible_witness_indices"], dtype=np.int64
        )
        parent = kwargs["parent_face_frontier"]
        iterator_calls.append((tuple(int(row) for row in possible), parent))
        counters = kwargs["counters"]
        assert isinstance(counters, _PadCounters)
        counters.swept_face_witness_stages += 1
        counters.swept_face_witnesses_materialized += 1
        yield int(possible[0]), np.asarray((0,), dtype=np.int64)

    original_groups = model._exact_plane_groups_v9
    group_calls = 0

    def controlled_groups(faces: np.ndarray) -> tuple[np.ndarray, ...]:
        nonlocal group_calls
        group_calls += 1
        if group_calls == 1:
            row = np.asarray((int(faces[0]),), dtype=np.int64)
            row.setflags(write=False)
            return tuple(row for _index in range(257))
        return original_groups(faces)

    def certified_free_rows(**kwargs: object) -> object:
        for witness_index, faces in kwargs["swept_batches"]:
            for face_index in faces:
                yield _PairIntervalClassification(
                    state=_PadSearchState.CERTIFIED_FREE,
                    witness_flat_index=int(witness_index),
                    object_face_index=int(face_index),
                    possible_phase_lower=float(kwargs["lower"]),
                    root=None,
                    reason="CONTROLLED_CERTIFIED_FREE",
                )

    monkeypatch.setattr(model, "_interval_geometry", controlled_geometry)
    monkeypatch.setattr(
        model, "_iter_complete_swept_face_batches_v9", controlled_batches
    )
    monkeypatch.setattr(model, "_exact_plane_groups_v9", controlled_groups)
    monkeypatch.setattr(
        model,
        "_iter_classify_witness_face_batches_parallel_v9",
        certified_free_rows,
    )
    counters = _PadCounters()
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
        + _FK_ERROR
        * (model.intersector.characteristic_length_m + hand_extent)
    )

    outcome = model._search_pad_first_contact_v9(
        prepared=prepared,
        q_start=q_start,
        direction=direction,
        maximum_parameter=maximum_parameter,
        object_from_hand=object_from_hand,
        spatial_error_bound_m=spatial_error,
        budget=_Budget(32),
        counters=counters,
        execution=_GeometryExecutionContext(),
    )

    assert outcome.state is _PadSearchState.CERTIFIED_FREE
    assert iterator_calls[0][0] == tuple(range(witness_count))
    assert iterator_calls[0][1] is None
    assert iterator_calls[1][0] == (later_witness,)
    assert iterator_calls[1][1] is None
    assert counters.staged_potential_root_temporal_deferrals == 1
    assert counters.staged_unmaterialized_witnesses >= witness_count - 1


def test_exact_plane_groups_merge_opposite_winding_without_cross_plane_merge() -> None:
    model, _hand, _pads = _planner(_box_model())
    faces = np.arange(
        len(model.canonical_object_face_vertices_m), dtype=np.int64
    )

    groups = model._exact_plane_groups_v9(faces)

    assert [row.tolist() for row in groups] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
    ]
    assert all(not row.flags.writeable for row in groups)
    assert sorted(int(value) for row in groups for value in row) == list(
        range(len(faces))
    )
    for row in groups:
        exact_keys = {
            model._exact_plane_key_for_face_v9(int(face_index))
            for face_index in row
        }
        assert len(exact_keys) == 1
    assert len(
        {
            model._exact_plane_key_for_face_v9(int(row[0]))
            for row in groups
        }
    ) == len(groups)
    fast_cache_size = len(model._fast_plane_bucket_key_cache)
    exact_cache_size = len(model._exact_plane_key_cache)
    repeated = model._exact_plane_groups_v9(faces)
    assert [row.tolist() for row in repeated] == [
        row.tolist() for row in groups
    ]
    assert len(model._fast_plane_bucket_key_cache) == fast_cache_size
    assert len(model._exact_plane_key_cache) == exact_cache_size


def test_fast_exact_dyadic_plane_key_equals_fraction_reference() -> None:
    model, _hand, _pads = _planner(_box_model())
    triangles = list(model.canonical_object_face_vertices_m)
    generator = np.random.default_rng(20260822)
    triangles.extend(generator.normal(size=(96, 3, 3)))

    checked = 0
    for triangle in triangles:
        try:
            fast = _exact_dyadic_plane_key(triangle)
            reference = _exact_dyadic_plane_key_fraction_reference(
                triangle
            )
        except RayClosureError:
            continue
        assert fast == reference
        checked += 1
    assert checked == len(triangles)


def test_shared_plane_batch_classification_matches_scalar_states_and_roots() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    faces = np.asarray(
        (11, 0, 10, 1, 3, 2, 5, 4, 7, 6, 9, 8),
        dtype=np.int64,
    )

    scalar_rows = tuple(
        model._classify_witness_face_pair_v9(
            prepared=prepared,
            witness_flat_index=0,
            object_face_index=int(face_index),
            q_start=q_start,
            direction=direction,
            lower=0.0,
            upper=maximum_parameter,
            object_from_hand=object_from_hand,
            counters=_PadCounters(),
            transform_cache=None,
        )
        for face_index in faces
    )
    batch_counters = _PadCounters()
    batch_rows = tuple(
        model._iter_classify_witness_face_batch_v9(
            prepared=prepared,
            witness_flat_index=0,
            object_face_indices=faces,
            q_start=q_start,
            direction=direction,
            lower=0.0,
            upper=maximum_parameter,
            object_from_hand=object_from_hand,
            counters=batch_counters,
            transform_cache=None,
        )
    )

    assert tuple(row.object_face_index for row in batch_rows) == tuple(faces)
    assert tuple(row.state for row in batch_rows) == tuple(
        row.state for row in scalar_rows
    )
    improved_root_count = 0
    for scalar, batch in zip(scalar_rows, batch_rows):
        assert batch.witness_flat_index == scalar.witness_flat_index
        assert batch.object_face_index == scalar.object_face_index
        assert batch.possible_phase_lower == scalar.possible_phase_lower
        assert (batch.root is None) == (scalar.root is None)
        if scalar.root is not None:
            assert batch.root is not None
            assert batch.root.pad_name == scalar.root.pad_name
            assert batch.root.witness_flat_index == scalar.root.witness_flat_index
            assert batch.root.pad_triangle_index == scalar.root.pad_triangle_index
            assert batch.root.witness_index == scalar.root.witness_index
            assert batch.root.object_face_index == scalar.root.object_face_index
            assert (
                batch.root.semantic_classification
                == scalar.root.semantic_classification
            )
            batch_certificate = batch.root.certificate
            scalar_certificate = scalar.root.certificate

            def assert_overlap(first, second) -> None:
                assert max(first.lower, second.lower) <= min(
                    first.upper, second.upper
                )
                assert first.strictly_positive == second.strictly_positive
                assert first.strictly_negative == second.strictly_negative

            batch_implicit = batch_certificate.implicit_root
            scalar_implicit = scalar_certificate.implicit_root
            for batch_bounds, scalar_bounds in (
                (batch_implicit.value_at_lower, scalar_implicit.value_at_lower),
                (batch_implicit.value_at_upper, scalar_implicit.value_at_upper),
                (batch_implicit.derivative, scalar_implicit.derivative),
                (batch_certificate.pad_approach, scalar_certificate.pad_approach),
                (
                    batch_certificate.path_local_free_side_approach,
                    scalar_certificate.path_local_free_side_approach,
                ),
                *zip(
                    batch_certificate.triangle_edge_halfspaces,
                    scalar_certificate.triangle_edge_halfspaces,
                ),
                *zip(
                    batch_certificate.position_object_m,
                    scalar_certificate.position_object_m,
                ),
            ):
                assert_overlap(batch_bounds, scalar_bounds)
            normalized_implicit = replace(
                batch_implicit,
                value_at_lower=scalar_implicit.value_at_lower,
                value_at_upper=scalar_implicit.value_at_upper,
                derivative=scalar_implicit.derivative,
            )
            normalized_certificate = replace(
                batch_certificate,
                implicit_root=normalized_implicit,
                triangle_edge_halfspaces=(
                    scalar_certificate.triangle_edge_halfspaces
                ),
                pad_approach=scalar_certificate.pad_approach,
                path_local_free_side_approach=(
                    scalar_certificate.path_local_free_side_approach
                ),
                position_object_m=scalar_certificate.position_object_m,
                bisection_iterations=(
                    scalar_certificate.bisection_iterations
                ),
            )
            assert normalized_certificate == scalar_certificate
            if (
                batch.root.certificate.bisection_iterations
                < scalar.root.certificate.bisection_iterations
            ):
                improved_root_count += 1
            assert batch.reason == scalar.reason
    assert improved_root_count >= 1
    assert batch_counters.interval_pair_evaluations == len(faces)
    assert batch_counters.actual_plane_root_evaluations == 6
    assert (
        batch_counters.actual_plane_root_evaluations
        < batch_counters.interval_pair_evaluations
    )
    assert batch_counters.batch_root_triangle_free_pairs == 1
    assert batch_counters.batch_root_triangle_uncertain_pairs == 1
    assert (
        batch_counters.root_interpolation_iterations
        + batch_counters.interval_newton_iterations
        >= 1
    )
    assert batch_counters.root_bisection_iterations <= 8


def test_parallel_shared_motion_plane_pipeline_preserves_order_and_states() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    faces = np.asarray(
        (11, 0, 10, 1, 3, 2, 5, 4, 7, 6, 9, 8),
        dtype=np.int64,
    )
    reference_rows = tuple(
        model._iter_classify_witness_face_batch_v9(
            prepared=prepared,
            witness_flat_index=0,
            object_face_indices=faces,
            q_start=q_start,
            direction=direction,
            lower=0.0,
            upper=maximum_parameter,
            object_from_hand=object_from_hand,
            counters=_PadCounters(),
            transform_cache=None,
        )
    )
    counters = _PadCounters()
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_rows = tuple(
            model._iter_classify_witness_face_batches_parallel_v9(
                prepared=prepared,
                swept_batches=((0, faces),),
                q_start=q_start,
                direction=direction,
                lower=0.0,
                upper=maximum_parameter,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                plane_root_executor=executor,
                plane_root_worker_local=local(),
                cache_enabled=True,
            )
        )

    assert tuple(row.object_face_index for row in parallel_rows) == tuple(faces)
    assert tuple(row.state for row in parallel_rows) == tuple(
        row.state for row in reference_rows
    )
    assert tuple(row.root is None for row in parallel_rows) == tuple(
        row.root is None for row in reference_rows
    )
    for reference, parallel in zip(reference_rows, parallel_rows):
        assert parallel.witness_flat_index == reference.witness_flat_index
        assert parallel.possible_phase_lower == reference.possible_phase_lower
        if reference.root is None:
            continue
        assert parallel.root is not None
        assert (
            parallel.root.semantic_classification
            == reference.root.semantic_classification
        )
        reference_phase = reference.root.certificate.phase
        parallel_phase = parallel.root.certificate.phase
        assert max(reference_phase.lower, parallel_phase.lower) <= min(
            reference_phase.upper, parallel_phase.upper
        )
    assert counters.interval_pair_evaluations == len(faces)
    assert 0 < counters.parallel_plane_root_tasks <= 6
    assert counters.actual_plane_root_evaluations == (
        counters.parallel_plane_root_tasks
    )
    assert counters.shared_plane_gate_roots >= 1
    assert counters.pre_root_spatial_enclosure_groups >= 1
    assert counters.pre_root_spatial_free_pairs >= 1


def test_parent_pair_inheritance_is_fail_closed_at_child_boundary() -> None:
    free = _PairIntervalClassification(
        state=_PadSearchState.CERTIFIED_FREE,
        witness_flat_index=3,
        object_face_index=7,
        possible_phase_lower=0.0,
        root=None,
        reason="PARENT_FREE",
    )
    unresolved = replace(
        free,
        state=_PadSearchState.UNRESOLVED,
        reason="PARENT_UNRESOLVED",
    )

    class _Root:
        class _Certificate:
            phase = IntervalBounds(0.4, 0.6)

        certificate = _Certificate()

    root = replace(
        free,
        state=_PadSearchState.CERTIFIED_ROOT,
        root=_Root(),
        reason="PARENT_ROOT",
    )
    decide = RayClosureSurfaceModel._parent_pair_inheritance_for_child_v9

    assert decide(
        free, child_lower=0.2, child_upper=0.5
    ) is _ParentPairInheritance.PRUNE_PARENT_CERTIFIED_FREE
    assert decide(
        unresolved, child_lower=0.2, child_upper=0.5
    ) is _ParentPairInheritance.RECOMPUTE
    assert decide(
        root, child_lower=0.0, child_upper=0.3
    ) is _ParentPairInheritance.PRUNE_PARENT_ROOT_DISJOINT
    assert decide(
        root, child_lower=0.3, child_upper=0.7
    ) is _ParentPairInheritance.REUSE_PARENT_ROOT
    assert decide(
        root, child_lower=0.5, child_upper=0.8
    ) is _ParentPairInheritance.RECOMPUTE


def test_parallel_pipeline_reuses_exact_parent_root_without_reordering() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    faces = np.arange(len(model.canonical_object_face_vertices_m), dtype=np.int64)
    baseline_counters = _PadCounters()
    with ThreadPoolExecutor(max_workers=4) as executor:
        baseline = tuple(
            model._iter_classify_witness_face_batches_parallel_v9(
                prepared=prepared,
                swept_batches=((0, faces),),
                q_start=q_start,
                direction=direction,
                lower=0.0,
                upper=maximum_parameter,
                object_from_hand=object_from_hand,
                counters=baseline_counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                plane_root_executor=executor,
                plane_root_worker_local=local(),
                cache_enabled=True,
            )
        )
    cached = {
        (row.witness_flat_index, row.object_face_index): row
        for row in baseline
        if row.state is _PadSearchState.CERTIFIED_ROOT
    }
    assert cached

    reused_counters = _PadCounters()
    with ThreadPoolExecutor(max_workers=4) as executor:
        reused = tuple(
            model._iter_classify_witness_face_batches_parallel_v9(
                prepared=prepared,
                swept_batches=((0, faces),),
                q_start=q_start,
                direction=direction,
                lower=0.0,
                upper=maximum_parameter,
                object_from_hand=object_from_hand,
                counters=reused_counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                plane_root_executor=executor,
                plane_root_worker_local=local(),
                cache_enabled=True,
                preclassified_pairs=cached,
            )
        )

    assert tuple(row.object_face_index for row in reused) == tuple(faces)
    assert tuple(row.state for row in reused) == tuple(
        row.state for row in baseline
    )
    assert tuple(row.root is None for row in reused) == tuple(
        row.root is None for row in baseline
    )
    assert reused_counters.parent_certified_root_pair_reuses == len(cached)
    assert reused_counters.actual_plane_root_evaluations < (
        baseline_counters.actual_plane_root_evaluations
    )


def test_large_exact_batch_defers_before_any_root_submission() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    faces = np.arange(len(model.canonical_object_face_vertices_m), dtype=np.int64)
    repeated_batches = tuple((0, faces) for _index in range(128))
    counters = _PadCounters()

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = tuple(
            model._iter_classify_witness_face_batches_parallel_v9(
                prepared=prepared,
                swept_batches=repeated_batches,
                q_start=q_start,
                direction=direction,
                lower=0.0,
                upper=maximum_parameter,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                plane_root_executor=executor,
                plane_root_worker_local=local(),
                cache_enabled=True,
            )
        )

    assert len(rows) == 1
    assert rows[0].state is _PadSearchState.UNRESOLVED
    assert rows[0].reason.startswith(
        "LARGE_EXACT_ROOT_BATCH_DEFERRED_TO_TEMPORAL_CHILD:"
    )
    assert counters.large_exact_batch_temporal_deferrals == 1
    assert counters.large_exact_batch_deferred_root_groups > 256
    assert counters.actual_plane_root_evaluations == 0
    assert counters.parallel_plane_root_tasks == 0


def test_pre_root_spatial_cull_skips_exact_root_for_outside_triangle() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    coplanar_faces = np.asarray((10, 11), dtype=np.int64)
    reference_rows = tuple(
        model._iter_classify_witness_face_batch_v9(
            prepared=prepared,
            witness_flat_index=0,
            object_face_indices=coplanar_faces,
            q_start=q_start,
            direction=direction,
            lower=0.0,
            upper=maximum_parameter,
            object_from_hand=object_from_hand,
            counters=_PadCounters(),
            transform_cache=None,
        )
    )
    outside_rows = tuple(
        row
        for row in reference_rows
        if row.state is _PadSearchState.CERTIFIED_FREE and row.root is None
    )
    assert len(outside_rows) == 1
    outside_face = outside_rows[0].object_face_index

    counters = _PadCounters()
    with ThreadPoolExecutor(max_workers=4) as executor:
        optimized_rows = tuple(
            model._iter_classify_witness_face_batches_parallel_v9(
                prepared=prepared,
                swept_batches=(
                    (0, np.asarray((outside_face,), dtype=np.int64)),
                ),
                q_start=q_start,
                direction=direction,
                lower=0.0,
                upper=maximum_parameter,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=(
                    model.interval_kinematics.new_link_transform_cache()
                ),
                plane_root_executor=executor,
                plane_root_worker_local=local(),
                cache_enabled=True,
            )
        )

    assert len(optimized_rows) == 1
    assert optimized_rows[0].state is _PadSearchState.CERTIFIED_FREE
    assert optimized_rows[0].root is None
    assert counters.pre_root_spatial_enclosure_groups == 1
    assert counters.pre_root_spatial_free_pairs == 1
    assert counters.pre_root_spatial_fully_free_groups == 1
    assert counters.parallel_plane_root_tasks == 0
    assert counters.actual_plane_root_evaluations == 0


@pytest.mark.parametrize("start_x,end_x", ((0.0, 1.0), (1.0, 0.0)))
def test_pre_root_spatial_enclosure_contains_linear_root_and_point(
    start_x: float,
    end_x: float,
) -> None:
    epsilon = 1.0e-12
    velocity_x = end_x - start_x

    def motion(
        phase_lower: float,
        phase_upper: float,
        position_x_lower: float,
        position_x_upper: float,
        whole: bool,
    ) -> IntervalPointMotionBatch:
        if whole:
            position_lower = np.asarray(
                ((min(start_x, end_x) - epsilon, -epsilon, -epsilon),)
            )
            position_upper = np.asarray(
                ((max(start_x, end_x) + epsilon, epsilon, epsilon),)
            )
        else:
            position_lower = np.asarray(
                ((position_x_lower, -epsilon, -epsilon),)
            )
            position_upper = np.asarray(
                ((position_x_upper, epsilon, epsilon),)
            )
        velocity_lower = np.asarray(
            ((velocity_x - epsilon, -epsilon, -epsilon),)
        )
        velocity_upper = np.asarray(
            ((velocity_x + epsilon, epsilon, epsilon),)
        )
        zeros = np.zeros((1, 3), dtype=np.float64)
        return IntervalPointMotionBatch(
            phase=IntervalBounds(phase_lower, phase_upper),
            position_lower_object_m=position_lower,
            position_upper_object_m=position_upper,
            velocity_lower_object_m_per_unit=velocity_lower,
            velocity_upper_object_m_per_unit=velocity_upper,
            acceleration_lower_object_m_per_unit_squared=zeros,
            acceleration_upper_object_m_per_unit_squared=zeros,
            method_id=BATCH_POINT_MOTION_METHOD_ID,
            decimal_precision=80,
        )

    whole_motion = motion(0.0, 1.0, 0.0, 0.0, True)
    lower_motion = motion(
        0.0,
        0.0,
        start_x - epsilon,
        start_x + epsilon,
        False,
    )
    upper_motion = motion(
        1.0,
        1.0,
        end_x - epsilon,
        end_x + epsilon,
        False,
    )
    derivative = np.asarray((velocity_x,), dtype=np.float64)
    lower_value = np.asarray((start_x - 0.5,), dtype=np.float64)
    upper_value = np.asarray((end_x - 0.5,), dtype=np.float64)
    enclosure = (
        RayClosureSurfaceModel.
        _certified_monotone_root_position_enclosures_v9(
            phase_lower=0.0,
            phase_upper=1.0,
            derivative_lower=derivative - epsilon,
            derivative_upper=derivative + epsilon,
            lower_value_lower=lower_value - epsilon,
            lower_value_upper=lower_value + epsilon,
            upper_value_lower=upper_value - epsilon,
            upper_value_upper=upper_value + epsilon,
            whole_motion=whole_motion,
            lower_motion=lower_motion,
            upper_motion=upper_motion,
            eligible=np.asarray((True,)),
        )
    )

    assert enclosure.valid.tolist() == [True]
    assert enclosure.phase_lower[0] <= 0.5 <= enclosure.phase_upper[0]
    assert (
        enclosure.position_lower_object_m[0, 0]
        <= 0.5
        <= enclosure.position_upper_object_m[0, 0]
    )
    assert np.all(enclosure.position_lower_object_m[0, 1:] <= 0.0)
    assert np.all(enclosure.position_upper_object_m[0, 1:] >= 0.0)
    assert enclosure.phase_upper[0] - enclosure.phase_lower[0] < 1.0e-8
    assert not enclosure.valid.flags.writeable
    assert not enclosure.position_lower_object_m.flags.writeable


def test_direct_affine_root_bounds_contain_mixed_slope_samples() -> None:
    slopes = np.asarray(
        (
            (1.0, -2.0, 0.125),
            (-0.75, 3.0, -1.5),
            (2.5, -0.25, 0.0),
        ),
        dtype=np.float64,
    )
    intercepts = np.asarray(
        (
            (-0.4, 0.8, -0.1),
            (0.2, -1.1, 0.7),
            (-0.9, 0.05, -0.3),
        ),
        dtype=np.float64,
    )
    root_lower = np.asarray((0.21, 0.47, 0.69), dtype=np.float64)
    root_upper = np.asarray((0.29, 0.53, 0.76), dtype=np.float64)
    epsilon = 1.0e-13
    lower_values = intercepts
    upper_values = intercepts + slopes
    result_lower, result_upper = (
        RayClosureSurfaceModel.
        _certified_affine_values_at_root_from_endpoints_v9(
            phase_lower=0.0,
            phase_upper=1.0,
            root_phase_lower=root_lower,
            root_phase_upper=root_upper,
            lower_value_lower=lower_values - epsilon,
            lower_value_upper=lower_values + epsilon,
            upper_value_lower=upper_values - epsilon,
            upper_value_upper=upper_values + epsilon,
            derivative_lower=slopes - epsilon,
            derivative_upper=slopes + epsilon,
        )
    )

    for row in range(len(slopes)):
        for phase in np.linspace(root_lower[row], root_upper[row], 17):
            actual = intercepts[row] + slopes[row] * phase
            assert np.all(result_lower[row] <= actual)
            assert np.all(actual <= result_upper[row])
    assert not result_lower.flags.writeable
    assert not result_upper.flags.writeable


def test_second_order_affine_chord_bounds_contain_quadratic_samples() -> None:
    quadratic = np.asarray(
        (
            (1.25, -0.8, 0.15),
            (-1.75, 0.6, -0.3),
            (0.4, 1.1, -0.7),
        ),
        dtype=np.float64,
    )
    linear = np.asarray(
        (
            (-0.5, 0.9, 0.2),
            (1.2, -0.4, 0.75),
            (-0.3, 0.1, 1.4),
        ),
        dtype=np.float64,
    )
    intercept = np.asarray(
        (
            (0.1, -0.2, 0.8),
            (-0.6, 0.5, 0.05),
            (0.7, -0.9, -0.1),
        ),
        dtype=np.float64,
    )
    root_lower = np.asarray((0.17, 0.41, 0.73), dtype=np.float64)
    root_upper = np.asarray((0.26, 0.58, 0.91), dtype=np.float64)
    epsilon = 1.0e-13
    lower_values = intercept
    upper_values = quadratic + linear + intercept
    result_lower, result_upper = (
        RayClosureSurfaceModel.
        _certified_second_order_affine_chord_root_bounds_v9(
            phase_lower=0.0,
            phase_upper=1.0,
            root_phase_lower=root_lower,
            root_phase_upper=root_upper,
            lower_value_lower=lower_values - epsilon,
            lower_value_upper=lower_values + epsilon,
            upper_value_lower=upper_values - epsilon,
            upper_value_upper=upper_values + epsilon,
            second_derivative_lower=2.0 * quadratic - epsilon,
            second_derivative_upper=2.0 * quadratic + epsilon,
        )
    )

    for row in range(len(quadratic)):
        for phase in np.linspace(root_lower[row], root_upper[row], 33):
            actual = (
                quadratic[row] * phase * phase
                + linear[row] * phase
                + intercept[row]
            )
            assert np.all(result_lower[row] <= actual)
            assert np.all(actual <= result_upper[row])
    assert not result_lower.flags.writeable
    assert not result_upper.flags.writeable


def test_possible_frontier_coalesces_exact_leaf_queries_without_semantic_change() -> None:
    model, hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    midpoint = 0.5 * maximum_parameter
    execution = _GeometryExecutionContext(
        cache_enabled=True,
        verify_full_nearest=True,
    )
    states, state_key = model._cached_witness_states(
        prepared,
        q_start + midpoint * direction,
        direction,
        object_from_hand,
        execution,
    )
    speed_bounds = model._witness_speed_bounds(
        prepared,
        q_start,
        direction,
        maximum_parameter,
    )
    counters = _PadCounters()
    geometry = model._interval_geometry(
        prepared=prepared,
        states=states,
        state_key=state_key,
        enclosure_radii_m=speed_bounds * (0.5 * maximum_parameter),
        spatial_error_bound_m=(
            model.intersector.distance_error_bound_m
            + model.distance_bvh.aabb_error_bound_m
        ),
        counters=counters,
        execution=execution,
    )

    assert np.any(geometry.possible)
    assert execution.stats.exact_nearest_witness_queries <= len(states)
    assert execution.stats.nearest_batch_cache_misses <= 2
    assert execution.stats.reference_shadow_witness_queries == len(states)


def test_pre_nearest_aabb_prefilter_preserves_exact_possible_set() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    states = model._witness_states(
        prepared,
        q_start,
        direction,
        object_from_hand,
    )
    controlled_positions = np.array(
        states.positions_object_m, copy=True
    )
    controlled_positions[:] = np.asarray((10.0, 10.0, 10.0))
    controlled_positions[0] = model.canonical_object_face_vertices_m[0, 0]
    controlled_positions.setflags(write=False)
    controlled_states = replace(
        states,
        positions_object_m=controlled_positions,
    )
    full = model.distance_bvh.nearest_many(controlled_positions)
    full_possible = full.distances_m <= 0.0
    assert np.count_nonzero(full_possible) == 1

    counters = _PadCounters()
    execution = _GeometryExecutionContext(
        cache_enabled=True,
        verify_full_nearest=True,
    )
    geometry = model._interval_geometry(
        prepared=prepared,
        states=controlled_states,
        state_key=("H84_CONTROLLED_PARTIAL_OVERLAP",),
        enclosure_radii_m=np.zeros(len(controlled_states)),
        spatial_error_bound_m=0.0,
        counters=counters,
        execution=execution,
    )

    assert np.array_equal(geometry.possible, full_possible)
    assert np.array_equal(
        geometry.nearest_face_indices[geometry.possible],
        full.face_indices[full_possible],
    )
    assert geometry.minimum_free_margin_m is None
    assert counters.pre_nearest_aabb_witness_tests == len(controlled_states)
    assert (
        counters.pre_nearest_aabb_certified_free_witnesses
        == len(controlled_states) - 1
    )
    assert counters.pre_nearest_aabb_exact_survivors == 1
    assert counters.pre_nearest_aabb_fast_paths == 1
    assert counters.pre_nearest_aabb_fallbacks == 0
    assert execution.stats.exact_nearest_witness_queries == 1
    assert execution.stats.reference_shadow_witness_queries == len(
        controlled_states
    )


def test_exact_state_and_batch_nearest_caches_reuse_identical_geometry() -> None:
    model, _hand, _pads = _planner(_box_model())
    q_start, target, rotation = model._decode(_PARAMETERS)
    transform_result = model._object_from_hand(q_start, target, rotation)
    assert transform_result is not None
    object_from_hand, _hand_extent = transform_result
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    midpoint = 0.5 * maximum_parameter
    q_midpoint = q_start + midpoint * direction
    execution = _GeometryExecutionContext(cache_enabled=True)

    first_states, first_key = model._cached_witness_states(
        prepared,
        q_midpoint,
        direction,
        object_from_hand,
        execution,
    )
    q_with_unrelated_finger_changed = np.array(q_midpoint, copy=True)
    unrelated_index = next(
        index
        for index in range(len(q_midpoint))
        if index not in prepared.relevant_joint_indices
    )
    q_with_unrelated_finger_changed[unrelated_index] += 0.25
    second_states, second_key = model._cached_witness_states(
        prepared,
        q_with_unrelated_finger_changed,
        direction,
        object_from_hand,
        execution,
    )
    direct_recomputation = model._witness_states(
        prepared,
        q_with_unrelated_finger_changed,
        direction,
        object_from_hand,
    )

    assert second_key == first_key
    assert second_states is first_states
    assert np.array_equal(
        second_states.positions_object_m,
        direct_recomputation.positions_object_m,
    )
    assert np.array_equal(
        second_states.velocities_object_per_unit,
        direct_recomputation.velocities_object_per_unit,
    )
    assert np.array_equal(
        second_states.pad_source_winding_normals_object,
        direct_recomputation.pad_source_winding_normals_object,
    )
    assert np.array_equal(second_states.leading, direct_recomputation.leading)
    assert execution.stats.witness_state_cache_misses == 1
    assert execution.stats.witness_state_cache_hits == 1

    speed_bounds = model._witness_speed_bounds(
        prepared,
        q_start,
        direction,
        maximum_parameter,
    )
    enclosure_radii = speed_bounds * (0.5 * maximum_parameter)
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
    )
    first_counters = _PadCounters()
    first_geometry = model._interval_geometry(
        prepared=prepared,
        states=first_states,
        state_key=first_key,
        enclosure_radii_m=enclosure_radii,
        spatial_error_bound_m=spatial_error,
        counters=first_counters,
        execution=execution,
    )
    second_counters = _PadCounters()
    second_geometry = model._interval_geometry(
        prepared=prepared,
        states=second_states,
        state_key=second_key,
        enclosure_radii_m=enclosure_radii,
        spatial_error_bound_m=spatial_error,
        counters=second_counters,
        execution=execution,
    )

    assert np.array_equal(first_geometry.possible, second_geometry.possible)
    assert (
        first_geometry.minimum_free_margin_m
        == second_geometry.minimum_free_margin_m
    )
    assert first_counters == second_counters
    assert execution.stats.nearest_batch_cache_misses > 0
    assert execution.stats.nearest_batch_cache_hits > 0

    changed_relevant_q = np.array(q_midpoint, copy=True)
    changed_relevant_q[prepared.relevant_joint_indices[0]] += 0.125
    model._cached_witness_states(
        prepared,
        changed_relevant_q,
        direction,
        object_from_hand,
        execution,
    )
    assert len(execution.witness_state_cache) == 1
    assert execution.nearest_batch_cache == {}


def test_real_full_pad_hierarchy_prunes_only_full_nearest_equivalent_witnesses() -> None:
    model, hand = _real_hand_structural_planner()
    lower, upper = hand.joint_limit_vectors()
    q_start = lower + 0.25 * (upper - lower)
    prepared = model.prepared_pads[0]
    direction = model.closing_directions_physical[0]
    maximum_parameter = model._maximum_path_parameter(q_start, direction)
    q_midpoint = q_start + 0.5 * maximum_parameter * direction
    object_from_hand = np.eye(4)
    object_from_hand[0, 3] = 10.0
    optimized_context = _GeometryExecutionContext(cache_enabled=True)
    states, state_key = model._cached_witness_states(
        prepared,
        q_midpoint,
        direction,
        object_from_hand,
        optimized_context,
    )
    speed_bounds = model._witness_speed_bounds(
        prepared,
        q_start,
        direction,
        maximum_parameter,
    )
    enclosure_radii = speed_bounds * (0.5 * maximum_parameter)
    spatial_error = (
        model.intersector.distance_error_bound_m
        + model.distance_bvh.aabb_error_bound_m
    )
    optimized_counters = _PadCounters()
    optimized = model._interval_geometry(
        prepared=prepared,
        states=states,
        state_key=state_key,
        enclosure_radii_m=enclosure_radii,
        spatial_error_bound_m=spatial_error,
        counters=optimized_counters,
        execution=optimized_context,
    )
    reference_context = _GeometryExecutionContext(
        cache_enabled=False,
        verify_full_nearest=True,
    )
    reference_counters = _PadCounters()
    reference = model._interval_geometry(
        prepared=prepared,
        states=states,
        state_key=state_key,
        enclosure_radii_m=enclosure_radii,
        spatial_error_bound_m=spatial_error,
        counters=reference_counters,
        execution=reference_context,
        _use_pre_nearest_aabb_prefilter=False,
    )

    assert np.array_equal(optimized.possible, reference.possible)
    assert optimized.minimum_free_margin_m == reference.minimum_free_margin_m
    assert replace(
        optimized_counters,
        pre_nearest_aabb_witness_tests=0,
        pre_nearest_aabb_certified_free_witnesses=0,
        pre_nearest_aabb_exact_survivors=0,
        pre_nearest_aabb_fast_paths=0,
        pre_nearest_aabb_fallbacks=0,
    ) == reference_counters
    assert optimized_counters.pre_nearest_aabb_witness_tests == len(states)
    assert (
        optimized_counters.pre_nearest_aabb_certified_free_witnesses
        == len(states)
    )
    assert optimized_counters.pre_nearest_aabb_exact_survivors == 0
    assert optimized_counters.pre_nearest_aabb_fast_paths == 0
    assert optimized_counters.pre_nearest_aabb_fallbacks == 1
    assert optimized_context.stats.witness_hierarchy_witnesses_pruned > 0
    assert (
        optimized_context.stats.exact_nearest_witness_queries
        < len(states)
    )
    assert optimized_context.stats.nearest_batch_cache_misses <= 2
    assert reference_context.stats.reference_shadow_witness_queries == len(states)


def test_first_finger_fail_closed_deterministically_skips_later_fingers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, hand, _pads = _planner(_box_model(forbid_positive_x=True))
    original_search = model._search_pad_first_contact_v9
    searched_pads: list[str] = []

    def counted_search(**kwargs: object) -> object:
        prepared = kwargs["prepared"]
        searched_pads.append(prepared.verified.name)
        return original_search(**kwargs)

    monkeypatch.setattr(model, "_search_pad_first_contact_v9", counted_search)
    execution = _GeometryExecutionContext(
        cache_enabled=True,
        verify_full_nearest=True,
    )
    result = model._evaluate_unit_parameters_with_execution(
        _PARAMETERS,
        hand,
        execution=execution,
    )

    assert result.candidate is None
    assert result.audit.failure_reason == "FORBIDDEN_SEMANTIC_FIRST_CONTACT:pad_a"
    assert searched_pads == ["pad_a"]
    assert execution.stats.fail_closed_fingers_skipped == 2
