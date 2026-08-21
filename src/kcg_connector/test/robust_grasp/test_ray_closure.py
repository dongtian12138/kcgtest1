from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

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
    _Budget,
    _FK_ERROR,
    _GeometryExecutionContext,
    _PadCounters,
    _PadSearchState,
    _PointTriangleDistanceBvh,
    _prepare_pad,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DISPLAY_APPROXIMATION_ROLE,
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
    assert budget.used == 1

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


def test_v9_earlier_triangle_boundary_blocks_later_certified_root() -> None:
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
    assert counters.certified_contact_roots >= 1
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
        _q_start: np.ndarray, _rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        point = np.zeros(3, dtype=np.float64)
        return point, point.copy()

    monkeypatch.setattr(model, "_placement_coordinate_bounds", degenerate_bounds)
    degenerate = model.evaluate_unit_parameters(_PARAMETERS, hand)
    assert degenerate.candidate is None
    assert degenerate.audit.failure_reason is not None
    assert "zero-width placement axes" in degenerate.audit.failure_reason

    def empty_bounds(
        _q_start: np.ndarray, _rotation: np.ndarray
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
    )

    assert np.array_equal(optimized.possible, reference.possible)
    assert optimized.minimum_free_margin_m == reference.minimum_free_margin_m
    assert optimized_counters == reference_counters
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
