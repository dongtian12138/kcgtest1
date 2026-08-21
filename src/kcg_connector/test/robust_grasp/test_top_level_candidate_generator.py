from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)
from kcg_connector.grasp.robust.generation_checkpoint import (
    CanonicalCheckpointCodec,
    CheckpointLifecycle,
    GenerationCheckpointError,
    GenerationCheckpointRecoveryBlocked,
    GenerationCheckpointStore,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DISPLAY_APPROXIMATION_ROLE,
    IMPLICIT_ROOT_FEATURE_TYPE,
    IMPLICIT_ROOT_METHOD_ID,
    METHOD_ID as INTERVAL_METHOD_ID,
    CertifiedImplicitRoot,
    IntervalBounds,
    IntervalTransverseRootCertificate,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CertifiedContactFeatureRoot,
    CertifiedSequentialClosurePolicy,
    DisplayOnlyGraspProposal,
    METHOD_ID as V9_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_BINDING_UNBOUND_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX as V9_PARAMETER_LAYOUT_PREFIX,
    PossibleFirstContactSet,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
)
from kcg_connector.grasp.robust.surface_anchored_closure import (
    FIXED_ANCHOR_METHOD_ID,
    FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
    FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    ALLOWED_TOTAL_ATTEMPT_BUDGETS,
    FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION,
    FROZEN_V9_PARAMETER_DIMENSION,
    LANE_SPECS,
    LOCAL_REFINEMENT_EVALUATION_BUDGET,
    MAIN_TOTAL_ATTEMPT_BUDGET,
    AttemptStatus,
    CandidateLane,
    TopLevelCandidateGenerator,
    TopLevelCandidateGeneratorError,
    canonicalize_v9_parameters,
)


PAD_NAMES = ("pad_a", "pad_b", "pad_c")
PRESHAPE_NAMES = ("f1j1",)
V9_LAYOUT = V9_PARAMETER_LAYOUT_PREFIX + (
    "preshape_joint_unit:f1j1",
)
FIXED_LAYOUT = FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX + (
    "preshape_joint_unit:f1j1",
)


@dataclass(frozen=True)
class _Audit:
    failure_reason: str | None
    tag: str
    method_id: str = V9_METHOD_ID
    closure_parameter_domain_id: str = CLOSURE_PARAMETER_DOMAIN_ID
    parameter_domain_id: str = FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    parameter_layout: tuple[str, ...] = V9_LAYOUT
    preshape_joint_names: tuple[str, ...] = PRESHAPE_NAMES
    pad_order: tuple[str, ...] = PAD_NAMES
    full_verified_pad_mesh_used: bool = True
    pad_face_subset_input_allowed: bool = False
    parameters_unit: tuple[float, ...] | None = None
    anchor_pad_name: str | None = None
    delegated_volume_parameters_unit: tuple[float, ...] | None = None
    model_binding_complete: bool = False
    model_binding_status: str = MODEL_BINDING_UNBOUND_STATUS
    object_geometry_sha256: str = "UNBOUND_SYNTHETIC"
    model_contract_sha256: str = "UNBOUND_SYNTHETIC"
    pad_geometry_sha256: tuple[str, ...] = ()
    pad_runtime_geometry_sha256: tuple[str, ...] = ()
    pad_link_names: tuple[str, ...] = ()
    closing_directions_physical: tuple[tuple[float, ...], ...] = ()
    model_contract_canonical_json: str = ""
    independent_actuation_supports: tuple[tuple[str, ...], ...] = ()
    candidate_role: str = "NO_CANDIDATE"
    candidate_exact_contact_endpoint_certified: bool = False
    display_approximation_role: str = DISPLAY_APPROXIMATION_ROLE
    possible_first_contact_set_sha256: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Proposal:
    v9_parameters_unit: tuple[float, ...] | None
    audit: _Audit


@dataclass(frozen=True)
class _V9Evaluation:
    candidate: object | None
    audit: _Audit
    sequential_closure_policy: object | None = None
    possible_first_contact_sets: tuple[object, ...] = ()
    display_only_proposal: object | None = None


def _grasp_candidate(
    pad_names: tuple[str, str, str] = PAD_NAMES,
) -> GraspCandidate:
    contacts = tuple(
        PlannedPadContact(
            pad_name=name,
            position_object_m=(float(index), 0.0, 0.0),
            path_local_free_side_normal_object=(1.0, 0.0, 0.0),
        )
        for index, name in enumerate(pad_names)
    )
    return GraspCandidate.from_matrix(
        object_from_hand=np.eye(4, dtype=np.float64),
        independent_joint_positions_rad=(0.0,),
        planned_pad_contacts=contacts,
        internal_normal_forces_n=(0.0, 0.0, 0.0),
    )


def _possible_contact_set(
    pad_name: str,
    ordinal: int,
) -> PossibleFirstContactSet:
    lower = 0.20 + 0.05 * ordinal
    upper = lower + 0.01
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
        value_at_lower=IntervalBounds(0.10, 0.20),
        value_at_upper=IntervalBounds(-0.20, -0.10),
        derivative=IntervalBounds(-2.0, -1.0),
        uniqueness_proven=True,
        display_approximation=lower + 0.5 * (upper - lower),
        display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
    )
    certificate = IntervalTransverseRootCertificate(
        implicit_root=implicit,
        triangle_edge_halfspaces=(
            IntervalBounds(0.10, 0.20),
            IntervalBounds(0.10, 0.20),
            IntervalBounds(0.10, 0.20),
        ),
        pad_approach=IntervalBounds(0.50, 1.00),
        path_local_free_side_approach=IntervalBounds(0.50, 1.00),
        object_source_winding_free_side_sign=1,
        position_object_m=(
            IntervalBounds(float(ordinal), float(ordinal) + 0.01),
            IntervalBounds(-0.01, 0.01),
            IntervalBounds(-0.01, 0.01),
        ),
        bisection_iterations=8,
        method_id=INTERVAL_METHOD_ID,
        decimal_precision=80,
    )
    return PossibleFirstContactSet.from_certified_roots(
        (
            CertifiedContactFeatureRoot(
                pad_name=pad_name,
                witness_flat_index=ordinal,
                pad_triangle_index=0,
                witness_index=ordinal,
                object_face_index=ordinal,
                semantic_classification=(
                    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
                ),
                certificate=certificate,
            ),
        )
    )


def _sequential_policy(
    v9: "_V9Contract",
) -> CertifiedSequentialClosurePolicy:
    contact_sets = tuple(
        _possible_contact_set(name, index)
        for index, name in enumerate(v9.pad_names)
    )
    return CertifiedSequentialClosurePolicy(
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        initial_independent_joint_positions_rad=(0.0, 0.0, 0.0, 0.5),
        independent_joint_names=("j0", "j1", "j2", "j3"),
        pad_order=v9.pad_names,
        independent_actuation_supports=(("j0",), ("j1",), ("j2",)),
        closing_directions_physical=v9.closing_directions_physical,
        possible_first_contact_sets=contact_sets,
        object_geometry_sha256=v9.object_geometry_sha256,
        model_contract_sha256=v9.model_contract_sha256,
    )


def _surface_audit(
    *,
    parameters: tuple[float, ...],
    anchor_pad_name: str,
    mapped: tuple[float, ...] | None,
    failure_reason: str | None = None,
    tag: str = "SURFACE",
) -> _Audit:
    return _Audit(
        failure_reason=failure_reason,
        tag=tag,
        method_id=FIXED_ANCHOR_METHOD_ID,
        parameter_layout=FIXED_LAYOUT,
        parameters_unit=parameters,
        anchor_pad_name=anchor_pad_name,
        delegated_volume_parameters_unit=mapped,
    )


class _V9Contract:
    method_id = V9_METHOD_ID
    closure_parameter_domain_id = CLOSURE_PARAMETER_DOMAIN_ID
    parameter_layout = V9_LAYOUT
    preshape_joint_names = PRESHAPE_NAMES

    def bind_contract(
        self, hand_model: object, pad_names: tuple[str, str, str]
    ) -> None:
        self.hand_model = hand_model
        self.pad_names = pad_names
        self.prepared_pads = tuple(
            SimpleNamespace(verified=SimpleNamespace(name=name))
            for name in pad_names
        )
        model_token = getattr(self, "model_token", "shared_model")
        self.model_binding_complete = True
        self.model_binding_status = MODEL_BINDING_COMPLETE_STATUS
        self.object_geometry_sha256 = hashlib.sha256(
            f"object:{model_token}".encode("utf-8")
        ).hexdigest()
        self.pad_geometry_sha256 = tuple(
            hashlib.sha256(f"source:{name}".encode("utf-8")).hexdigest()
            for name in pad_names
        )
        self.pad_runtime_geometry_sha256 = tuple(
            hashlib.sha256(f"runtime:{name}".encode("utf-8")).hexdigest()
            for name in pad_names
        )
        self.pad_link_names = tuple(f"{name}_link" for name in pad_names)
        self.closing_directions_physical = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
        document = {
            "schema": MODEL_CONTRACT_DIGEST_METHOD_ID,
            "object": {
                "geometry_sha256": self.object_geometry_sha256,
                "assembly_axis": [0.0.hex(), 0.0.hex(), 1.0.hex()],
                "assembly_axis_origin_m": [
                    0.0.hex(),
                    0.0.hex(),
                    0.0.hex(),
                ],
            },
            "task_frame": {
                "source": "TEST_PRE_REGISTERED_TASK_FRAME",
                "pre_registered_transverse_axis_object": [
                    1.0.hex(),
                    0.0.hex(),
                    0.0.hex(),
                ],
                "basis_object": [
                    [1.0.hex(), 0.0.hex(), 0.0.hex()],
                    [0.0.hex(), 1.0.hex(), 0.0.hex()],
                    [0.0.hex(), 0.0.hex(), 1.0.hex()],
                ],
            },
            "hand": {
                "base_link": "test_hand_base",
                "joint_order": ["j0", "j1", "j2", "j3"],
                "independent_joint_names": ["j0", "j1", "j2", "j3"],
                "joints_by_mapping_key": [],
                "independent_affine_limits": [],
                "finger_chains": [],
                "pad_mapping": [],
            },
            "verified_pads": [
                {
                    "name": name,
                    "finger_name": f"finger_{index}",
                    "link_name": self.pad_link_names[index],
                    "origin_xyz_m": [0.0.hex(), 0.0.hex(), 0.0.hex()],
                    "origin_rpy_rad": [0.0.hex(), 0.0.hex(), 0.0.hex()],
                    "coordinate_frame": "PAD_LOCAL",
                    "unit": "m",
                    "normal_force_capacity_n": 1.0.hex(),
                    "source_mesh_repository_relative_path": (
                        f"test/{name}.stl"
                    ),
                    "source_mesh_sha256": self.pad_geometry_sha256[index],
                    "source_mesh_byte_count": 1,
                    "runtime_geometry_sha256": (
                        self.pad_runtime_geometry_sha256[index]
                    ),
                    "vertex_count": 3,
                    "triangle_count": 1,
                }
                for index, name in enumerate(pad_names)
            ],
            "closure": {
                "closing_directions_unit": [
                    [float(value).hex() for value in row]
                    for row in self.closing_directions_physical
                ],
                "closing_directions_physical": [
                    [float(value).hex() for value in row]
                    for row in self.closing_directions_physical
                ],
                "independent_actuation_supports": [
                    ["j0"],
                    ["j1"],
                    ["j2"],
                ],
                "parameter_layout": list(V9_LAYOUT),
            },
            "ray_closure": {
                "method_id": V9_METHOD_ID,
                "closure_parameter_domain_id": CLOSURE_PARAMETER_DOMAIN_ID,
            },
            "interval_backend": {
                "method_id": "TEST_INTERVAL_BACKEND",
                "decimal_precision": 80,
                "maximum_root_bisection_iterations": 256,
            },
        }
        self.model_contract_canonical_json = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.model_contract_sha256 = hashlib.sha256(
            self.model_contract_canonical_json.encode("utf-8")
        ).hexdigest()

    def bound_audit(
        self,
        failure_reason: str | None,
        tag: str,
    ) -> _Audit:
        return _Audit(
            failure_reason=failure_reason,
            tag=tag,
            pad_order=self.pad_names,
            model_binding_complete=self.model_binding_complete,
            model_binding_status=self.model_binding_status,
            object_geometry_sha256=self.object_geometry_sha256,
            model_contract_sha256=self.model_contract_sha256,
            pad_geometry_sha256=self.pad_geometry_sha256,
            pad_runtime_geometry_sha256=self.pad_runtime_geometry_sha256,
            pad_link_names=self.pad_link_names,
            closing_directions_physical=(
                self.closing_directions_physical
            ),
            model_contract_canonical_json=(
                self.model_contract_canonical_json
            ),
        )


class _SurfaceContract:
    fixed_anchor_method_id = FIXED_ANCHOR_METHOD_ID
    fixed_anchor_parameter_domain_id = FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    fixed_anchor_parameter_layout = FIXED_LAYOUT

    def bind_contract(
        self,
        closure_model: object,
        hand_model: object,
        pad_names: tuple[str, str, str],
    ) -> None:
        self.closure_model = closure_model
        self.hand_model = hand_model
        self.prepared_pad_names = pad_names


class _AcceptingV9(_V9Contract):
    def __init__(self, *, model_token: str = "shared_model") -> None:
        self.model_token = model_token
        self.calls: list[tuple[float, ...]] = []
        self.hand_models: list[object] = []

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        return _V9Evaluation(
            candidate=_grasp_candidate(self.pad_names),
            audit=self.bound_audit(None, "V9_STATIC_ONLY"),
        )


class _PolicyV9(_AcceptingV9):
    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        policy = _sequential_policy(self)
        contact_sets = policy.possible_first_contact_sets
        audit = replace(
            self.bound_audit(
                REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
                "V9_CONTACT_RANGE_POLICY",
            ),
            independent_actuation_supports=(
                ("j0",),
                ("j1",),
                ("j2",),
            ),
            candidate_role=CANDIDATE_REPRESENTATIVE_ROLE,
            candidate_exact_contact_endpoint_certified=False,
            possible_first_contact_set_sha256=tuple(
                row.set_sha256 for row in contact_sets
            ),
        )
        return _V9Evaluation(
            candidate=None,
            audit=audit,
            sequential_closure_policy=policy,
            possible_first_contact_sets=contact_sets,
            display_only_proposal=DisplayOnlyGraspProposal(
                _grasp_candidate(self.pad_names)
            ),
        )


class _ProjectingSurface(_SurfaceContract):
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], str, object]] = []

    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: object,
    ) -> _Proposal:
        values = tuple(float(value) for value in parameters6)
        self.calls.append((values, anchor_pad_name, hand_model))
        mapped = values[:5]
        return _Proposal(
            v9_parameters_unit=mapped,
            audit=_surface_audit(
                parameters=values,
                anchor_pad_name=anchor_pad_name,
                mapped=mapped,
                tag=f"SURFACE:{anchor_pad_name}",
            ),
        )


class _ConstantSurface(_SurfaceContract):
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.calls: list[str] = []

    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: object,
    ) -> _Proposal:
        del hand_model
        parameters = tuple(float(value) for value in parameters6)
        self.calls.append(anchor_pad_name)
        return _Proposal(
            v9_parameters_unit=self.values,
            audit=_surface_audit(
                parameters=parameters,
                anchor_pad_name=anchor_pad_name,
                mapped=self.values,
                tag=f"CONSTANT:{anchor_pad_name}",
            ),
        )


class _RejectZeroV9(_AcceptingV9):
    FAILURE = "NO_FIRST_CONTACT_FOR_PAD:pad_b"

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        if values == (0.0, 0.0, 0.0, 0.0, 0.0):
            return _V9Evaluation(
                None,
                self.bound_audit(self.FAILURE, "V9_REJECTION"),
            )
        return _V9Evaluation(
            _grasp_candidate(self.pad_names),
            self.bound_audit(None, "V9_STATIC_ONLY"),
        )


def _generator(
    *,
    v9: object | None = None,
    surface: object | None = None,
    hand_model: object | None = None,
    pad_names: tuple[str, str, str] = PAD_NAMES,
) -> TopLevelCandidateGenerator:
    hand = {"synthetic_hand": 1} if hand_model is None else hand_model
    bound_v9 = _AcceptingV9() if v9 is None else v9
    if isinstance(bound_v9, _V9Contract):
        bound_v9.bind_contract(hand, pad_names)
    bound_surface = _ProjectingSurface() if surface is None else surface
    if isinstance(bound_surface, _SurfaceContract):
        bound_surface.bind_contract(bound_v9, hand, pad_names)
    return TopLevelCandidateGenerator(
        v9_evaluator=bound_v9,
        surface_proposer=bound_surface,
        hand_model=hand,
        anchor_pad_names=pad_names,
    )


def _attempts_for_lane(
    result: object, lane: CandidateLane
) -> tuple[object, ...]:
    return tuple(row for row in result.attempts if row.lane is lane)


def test_frozen_lane_allowlist_seeds_dimensions_and_main_budget() -> None:
    assert ALLOWED_TOTAL_ATTEMPT_BUDGETS == (128, 256, 512)
    assert MAIN_TOTAL_ATTEMPT_BUDGET == 256
    assert LOCAL_REFINEMENT_EVALUATION_BUDGET == 0
    assert FROZEN_V9_PARAMETER_DIMENSION == 5
    assert FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION == 6
    generator = _generator()
    assert generator.v9_parameter_layout == V9_LAYOUT
    assert generator.fixed_anchor_parameter_layout == FIXED_LAYOUT
    assert [row.lane for row in LANE_SPECS] == [
        CandidateLane.DIRECT_V9,
        CandidateLane.SURFACE_PAD_A,
        CandidateLane.SURFACE_PAD_B,
        CandidateLane.SURFACE_PAD_C,
    ]
    assert [row.dimension for row in LANE_SPECS] == [5, 6, 6, 6]
    assert [row.sobol_seed for row in LANE_SPECS] == [
        20260820,
        20260821,
        20260822,
        20260823,
    ]
    assert [row.anchor_pad_ordinal for row in LANE_SPECS] == [None, 0, 1, 2]


def test_four_independent_designs_are_interleaved_and_nested_prefixes(
) -> None:
    results = {
        budget: _generator().generate(budget)
        for budget in ALLOWED_TOTAL_ATTEMPT_BUDGETS
    }

    for budget, result in results.items():
        per_lane = budget // 4
        assert len(result.attempts) == budget
        assert result.attempts_per_lane == per_lane
        assert [row.lane for row in result.attempts[:8]] == [
            CandidateLane.DIRECT_V9,
            CandidateLane.SURFACE_PAD_A,
            CandidateLane.SURFACE_PAD_B,
            CandidateLane.SURFACE_PAD_C,
        ] * 2
        for index, row in enumerate(result.attempts):
            assert row.attempt_index == index
            assert row.lane_point_index == index // 4
        assert Counter(row.lane for row in result.attempts) == {
            lane: per_lane for lane in CandidateLane
        }

    for lane in CandidateLane:
        points_128 = tuple(
            row.sobol_parameters_unit
            for row in _attempts_for_lane(results[128], lane)
        )
        points_256 = tuple(
            row.sobol_parameters_unit
            for row in _attempts_for_lane(results[256], lane)
        )
        points_512 = tuple(
            row.sobol_parameters_unit
            for row in _attempts_for_lane(results[512], lane)
        )
        assert points_256[:32] == points_128
        assert points_512[:64] == points_256

    first_anchor_points = [
        _attempts_for_lane(results[128], lane)[0].sobol_parameters_unit
        for lane in (
            CandidateLane.SURFACE_PAD_A,
            CandidateLane.SURFACE_PAD_B,
            CandidateLane.SURFACE_PAD_C,
        )
    ]
    assert len(set(first_anchor_points)) == 3


def test_fixed_pad_lanes_call_only_the_declared_surface_protocol() -> None:
    surface = _ProjectingSurface()
    v9 = _AcceptingV9()
    hand = object()
    generator = _generator(v9=v9, surface=surface, hand_model=hand)

    result = generator.generate(128)

    assert Counter(anchor for _, anchor, _ in surface.calls) == {
        "pad_a": 32,
        "pad_b": 32,
        "pad_c": 32,
    }
    assert all(len(parameters) == 6 for parameters, _, _ in surface.calls)
    assert all(supplied_hand is hand for _, _, supplied_hand in surface.calls)
    assert all(
        len(row.sobol_parameters_unit) == 5 and row.anchor_pad_name is None
        for row in _attempts_for_lane(result, CandidateLane.DIRECT_V9)
    )
    for lane, pad_name in (
        (CandidateLane.SURFACE_PAD_A, "pad_a"),
        (CandidateLane.SURFACE_PAD_B, "pad_b"),
        (CandidateLane.SURFACE_PAD_C, "pad_c"),
    ):
        assert all(
            len(row.sobol_parameters_unit) == 6
            and row.anchor_pad_name == pad_name
            for row in _attempts_for_lane(result, lane)
        )
    assert not hasattr(generator, "register_proposer")
    assert not hasattr(generator, "register_lane")
    assert result.local_refinement_evaluation_budget == 0


def test_canonical_v9_identity_wraps_yaw_and_normalizes_signed_zero() -> None:
    def canonical(values: tuple[float, ...]) -> object:
        return canonicalize_v9_parameters(values, parameter_layout=V9_LAYOUT)

    from_one = canonical((1.0, -0.0, 0.0, -0.0, 0.0))
    from_negative_zero = canonicalize_v9_parameters(
        (-0.0, 0.0, -0.0, 0.0, -0.0), parameter_layout=V9_LAYOUT
    )

    assert from_one.values == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert from_one.exact_key == from_negative_zero.exact_key
    assert all(math.copysign(1.0, value) > 0.0 for value in from_one.values)
    assert canonical((-0.25, 0.0, 0.0, 0.0, 0.0)).values[0] == 0.75
    assert canonical((2.25, 0.0, 0.0, 0.0, 0.0)).values[0] == 0.25

    adjacent = math.nextafter(0.25, 1.0)
    assert canonical(
        (0.25, 0.0, 0.0, 0.0, 0.0)
    ).exact_key != canonical(
        (adjacent, 0.0, 0.0, 0.0, 0.0)
    ).exact_key

    with pytest.raises(TopLevelCandidateGeneratorError, match="exactly five"):
        canonical((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(TopLevelCandidateGeneratorError, match="finite"):
        canonical((math.nan, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(TopLevelCandidateGeneratorError, match="closed unit"):
        canonical((0.0, 1.01, 0.0, 0.0, 0.0))


def test_duplicates_and_v9_failure_keep_all_lineage_without_replacement(
) -> None:
    v9 = _RejectZeroV9()
    surface = _ConstantSurface((1.0, -0.0, 0.0, -0.0, 0.0))
    result = _generator(v9=v9, surface=surface).generate(128)

    assert len(result.attempts) == 128
    assert len(surface.calls) == 96
    assert result.v9_evaluation_count == 33
    assert len(v9.calls) == 33
    assert len(set(v9.calls)) == 33
    assert result.duplicate_attempt_count == 95
    assert result.proposal_failure_count == 0
    assert len(result.unique_v9_evaluations) == 33
    assert len(result.accepted_candidates) == 32

    rejected = next(
        row
        for row in result.unique_v9_evaluations
        if row.v9_parameters_unit == (0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert rejected.status is AttemptStatus.V9_REJECTED
    assert rejected.v9_failure_reason == _RejectZeroV9.FAILURE
    assert len(rejected.lineage) == 96
    duplicate_rows = tuple(
        row
        for row in result.attempts
        if row.status is AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
    )
    assert len(duplicate_rows) == 95
    assert all(row.duplicate_of_attempt_index == 1 for row in duplicate_rows)
    assert all(
        row.v9_failure_reason == _RejectZeroV9.FAILURE
        for row in duplicate_rows
    )
    assert all(
        row.failure_reason == _RejectZeroV9.FAILURE
        for row in duplicate_rows
    )


class _MissingAuditProposal:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.v9_parameters_unit = values


class _InvalidSurface(_SurfaceContract):
    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: object,
    ) -> object:
        del hand_model
        parameters = tuple(float(value) for value in parameters6)
        if anchor_pad_name == "pad_a":
            return _Proposal(
                None,
                _surface_audit(
                    parameters=parameters,
                    anchor_pad_name=anchor_pad_name,
                    mapped=None,
                    failure_reason="NO_CANONICAL_ANCHOR",
                    tag="PROPOSAL",
                ),
            )
        if anchor_pad_name == "pad_b":
            mapped = (0.0, 0.0, 0.0, 0.0, 1.01)
            return _Proposal(
                mapped,
                _surface_audit(
                    parameters=parameters,
                    anchor_pad_name=anchor_pad_name,
                    mapped=mapped,
                    tag="OUTSIDE_V9_DOMAIN",
                ),
            )
        return _MissingAuditProposal((0.0, 0.0, 0.0, 0.0, 0.0))


def test_invalid_surface_results_burn_attempts_and_never_reach_v9() -> None:
    v9 = _AcceptingV9()
    result = _generator(v9=v9, surface=_InvalidSurface()).generate(128)

    assert len(result.attempts) == 128
    assert result.proposal_failure_count == 96
    assert result.v9_evaluation_count == 32
    assert len(v9.calls) == 32
    assert Counter(row.status for row in result.attempts) == {
        AttemptStatus.STATIC_V9_ACCEPTED: 32,
        AttemptStatus.PROPOSAL_REJECTED: 32,
        AttemptStatus.PROPOSAL_V9_DOMAIN_REJECTED: 32,
        AttemptStatus.PROPOSAL_PROTOCOL_REJECTED: 32,
    }
    pad_a = _attempts_for_lane(result, CandidateLane.SURFACE_PAD_A)
    pad_b = _attempts_for_lane(result, CandidateLane.SURFACE_PAD_B)
    pad_c = _attempts_for_lane(result, CandidateLane.SURFACE_PAD_C)
    assert all(row.failure_reason == "NO_CANONICAL_ANCHOR" for row in pad_a)
    assert all(
        row.failure_reason.startswith("PROPOSAL_V9_DOMAIN_REJECTED:")
        for row in pad_b
    )
    assert all(
        row.failure_reason == "PROPOSAL_PROTOCOL_MISSING_OR_NULL_AUDIT"
        for row in pad_c
    )


class _InconsistentV9(_V9Contract):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        del parameters_unit, hand_model
        self.calls += 1
        return _V9Evaluation(
            candidate=_grasp_candidate(self.pad_names),
            audit=self.bound_audit(
                "MAXIMUM_SUBDIVISION_INTERVALS_EXHAUSTED",
                "V9",
            ),
        )


class _NullAuditV9(_V9Contract):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> dict[str, object | None]:
        del parameters_unit, hand_model
        self.calls += 1
        return {"candidate": _grasp_candidate(self.pad_names), "audit": None}


class _MissingFailureFieldV9(_AcceptingV9):
    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> dict[str, object]:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        return {"candidate": _grasp_candidate(self.pad_names), "audit": {}}


class _WrongCandidateTypeV9(_AcceptingV9):
    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        return _V9Evaluation(
            "NOT_A_GRASP_CANDIDATE",
            self.bound_audit(None, "V9"),
        )


class _WrongMethodAuditV9(_AcceptingV9):
    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        return _V9Evaluation(
            _grasp_candidate(self.pad_names),
            replace(
                self.bound_audit(None, "V9"),
                method_id="NOT_PRODUCTION_V9",
            ),
        )


class _WrongCandidatePadOrderV9(_AcceptingV9):
    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        values = tuple(float(value) for value in parameters_unit)
        self.calls.append(values)
        self.hand_models.append(hand_model)
        return _V9Evaluation(
            _grasp_candidate(("pad_b", "pad_a", "pad_c")),
            self.bound_audit(None, "V9"),
        )


class _ModelBindingAuditV9(_AcceptingV9):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.mutation = mutation

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> _V9Evaluation:
        result = super().evaluate_unit_parameters(
            parameters_unit,
            hand_model,
        )
        audit = result.audit
        if self.mutation == "missing":
            payload = dict(audit.__dict__)
            del payload["model_contract_sha256"]
            return _V9Evaluation(
                result.candidate,
                payload,  # type: ignore[arg-type]
            )
        if self.mutation == "synthetic_false":
            mutated: object = replace(
                audit,
                model_binding_complete=False,
            )
        elif self.mutation == "status":
            mutated = replace(
                audit,
                model_binding_status=MODEL_BINDING_UNBOUND_STATUS,
            )
        elif self.mutation == "object_sha256":
            mutated = replace(audit, object_geometry_sha256="0" * 64)
        elif self.mutation == "model_sha256":
            mutated = replace(audit, model_contract_sha256="0" * 64)
        elif self.mutation == "pad_source_sha256":
            rows = list(audit.pad_geometry_sha256)
            rows[0] = "0" * 64
            mutated = replace(audit, pad_geometry_sha256=tuple(rows))
        elif self.mutation == "pad_runtime_sha256":
            rows = list(audit.pad_runtime_geometry_sha256)
            rows[0] = "0" * 64
            mutated = replace(
                audit,
                pad_runtime_geometry_sha256=tuple(rows),
            )
        elif self.mutation == "pad_link_names":
            mutated = replace(
                audit,
                pad_link_names=("different_link",) + audit.pad_link_names[1:],
            )
        elif self.mutation == "closing_direction_bytes":
            rows = [list(row) for row in audit.closing_directions_physical]
            rows[0][1] = -0.0
            mutated = replace(
                audit,
                closing_directions_physical=tuple(
                    tuple(row) for row in rows
                ),
            )
        elif self.mutation == "noncanonical_json":
            mutated = replace(
                audit,
                model_contract_canonical_json=(
                    audit.model_contract_canonical_json + " "
                ),
            )
        elif self.mutation == "self_consistent_model_swap":
            document = json.loads(audit.model_contract_canonical_json)
            alternative_object_sha256 = hashlib.sha256(
                b"alternative-object-model"
            ).hexdigest()
            document["object"]["geometry_sha256"] = (
                alternative_object_sha256
            )
            canonical_json = json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            mutated = replace(
                audit,
                object_geometry_sha256=alternative_object_sha256,
                model_contract_sha256=hashlib.sha256(
                    canonical_json.encode("utf-8")
                ).hexdigest(),
                model_contract_canonical_json=canonical_json,
            )
        else:  # pragma: no cover - test construction invariant
            raise AssertionError(f"unknown mutation {self.mutation}")
        return _V9Evaluation(
            result.candidate,
            mutated,  # type: ignore[arg-type]
        )


class _MismatchedRequestSurface(_ProjectingSurface):
    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: object,
    ) -> _Proposal:
        proposal = super().propose_fixed_anchor(
            parameters6, anchor_pad_name, hand_model
        )
        audited = list(proposal.audit.parameters_unit or ())
        audited[0] = math.nextafter(audited[0], 1.0)
        return _Proposal(
            proposal.v9_parameters_unit,
            replace(proposal.audit, parameters_unit=tuple(audited)),
        )


class _RaisingSurface(_SurfaceContract):
    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: object,
    ) -> object:
        del parameters6, anchor_pad_name, hand_model
        raise RuntimeError("synthetic surface failure")


class _RaisingV9(_V9Contract):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_unit_parameters(
        self, parameters_unit: np.ndarray, hand_model: object
    ) -> object:
        del parameters_unit, hand_model
        self.calls += 1
        raise RuntimeError("synthetic V9 failure")


def test_v9_failure_reason_is_raw_and_inconsistent_candidate_fails_closed(
) -> None:
    v9 = _InconsistentV9()
    result = _generator(v9=v9).generate(128)

    assert v9.calls == result.v9_evaluation_count
    assert not result.accepted_candidates
    assert all(
        row.status is AttemptStatus.V9_PROTOCOL_REJECTED
        for row in result.unique_v9_evaluations
    )
    assert {
        row.v9_failure_reason for row in result.unique_v9_evaluations
    } == {"MAXIMUM_SUBDIVISION_INTERVALS_EXHAUSTED"}

    null_audit_v9 = _NullAuditV9()
    null_audit_result = _generator(v9=null_audit_v9).generate(128)
    assert null_audit_v9.calls == null_audit_result.v9_evaluation_count
    assert not null_audit_result.accepted_candidates
    assert {
        row.v9_failure_reason
        for row in null_audit_result.unique_v9_evaluations
    } == {"V9_PROTOCOL_REQUIRES_CANDIDATE_AND_NON_NULL_AUDIT"}


@pytest.mark.parametrize(
    "v9",
    (
        _MissingFailureFieldV9(),
        _WrongCandidateTypeV9(),
        _WrongMethodAuditV9(),
        _WrongCandidatePadOrderV9(),
    ),
)
def test_incomplete_or_wrong_v9_protocol_can_never_be_static_accepted(
    v9: object,
) -> None:
    result = _generator(v9=v9).generate(128)

    assert not result.accepted_candidates
    assert all(
        row.status is AttemptStatus.V9_PROTOCOL_REJECTED
        for row in result.unique_v9_evaluations
    )
    assert all(
        row.invocation_binding is None or row.candidate is None
        for row in result.unique_v9_evaluations
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "synthetic_false",
        "status",
        "object_sha256",
        "model_sha256",
        "pad_source_sha256",
        "pad_runtime_sha256",
        "pad_link_names",
        "closing_direction_bytes",
        "noncanonical_json",
        "self_consistent_model_swap",
    ),
)
def test_v9_model_binding_missing_synthetic_or_tampered_never_accepts(
    mutation: str,
) -> None:
    v9 = _ModelBindingAuditV9(mutation)
    result = _generator(v9=v9).generate(128)

    assert not result.accepted_candidates
    assert v9.calls
    assert all(
        row.status is AttemptStatus.V9_PROTOCOL_REJECTED
        for row in result.unique_v9_evaluations
    )
    assert all(
        row.v9_failure_reason is not None
        and row.v9_failure_reason.startswith(
            "V9_PROTOCOL_AUDIT_BINDING_REJECTED:"
        )
        for row in result.unique_v9_evaluations
    )


def test_surface_audit_must_bind_the_exact_six_dimensional_request() -> None:
    v9 = _AcceptingV9()
    result = _generator(
        v9=v9, surface=_MismatchedRequestSurface()
    ).generate(128)

    assert len(result.attempts) == 128
    assert len(v9.calls) == 32
    assert result.proposal_failure_count == 96
    assert Counter(row.status for row in result.attempts) == {
        AttemptStatus.STATIC_V9_ACCEPTED: 32,
        AttemptStatus.PROPOSAL_PROTOCOL_REJECTED: 96,
    }
    assert all(
        row.failure_reason is not None
        and "parameters_unit differs from the exact request"
        in row.failure_reason
        for lane in (
            CandidateLane.SURFACE_PAD_A,
            CandidateLane.SURFACE_PAD_B,
            CandidateLane.SURFACE_PAD_C,
        )
        for row in _attempts_for_lane(result, lane)
    )


def test_surface_and_v9_exceptions_burn_slots_without_replacement() -> None:
    surface_v9 = _AcceptingV9()
    surface_result = _generator(
        v9=surface_v9, surface=_RaisingSurface()
    ).generate(128)
    assert len(surface_result.attempts) == 128
    assert len(surface_v9.calls) == 32
    assert surface_result.proposal_failure_count == 96

    raising_v9 = _RaisingV9()
    v9_result = _generator(v9=raising_v9).generate(128)
    assert len(v9_result.attempts) == 128
    assert raising_v9.calls == v9_result.v9_evaluation_count
    assert not v9_result.accepted_candidates
    assert all(
        row.status is AttemptStatus.V9_EVALUATOR_EXCEPTION
        for row in v9_result.unique_v9_evaluations
    )


def test_contract_hash_binds_v9_model_and_has_no_formal_side_registry(
) -> None:
    current = _generator(hand_model={"object": "current"})
    held_out = _generator(hand_model={"object": "TE_held_out"})

    assert current.contract_hash_sha256 == held_out.contract_hash_sha256
    document = current.contract_document
    encoded = json.dumps(document, sort_keys=True)
    assert "current" not in encoded
    assert "TE_held_out" not in encoded
    assert document["main_total_attempt_budget"] == 256
    assert document["allowed_total_attempt_budgets"] == [128, 256, 512]
    assert document["sobol_design"]["scipy_version"] == scipy.__version__
    assert document["v9_certifier"]["method_id"] == V9_METHOD_ID
    assert document["v9_certifier"]["parameter_domain_id"] == (
        CLOSURE_PARAMETER_DOMAIN_ID
    )
    assert tuple(document["v9_certifier"]["parameter_layout"]) == V9_LAYOUT
    assert document["v9_certifier"]["model_binding_complete"] is True
    assert document["v9_certifier"]["model_binding_status"] == (
        MODEL_BINDING_COMPLETE_STATUS
    )
    assert document["v9_certifier"]["model_contract_digest_method_id"] == (
        MODEL_CONTRACT_DIGEST_METHOD_ID
    )
    assert document["v9_certifier"]["model_contract_sha256"] == (
        current.v9_model_contract_sha256
    )
    assert document["fixed_anchor_mapper"]["method_id"] == (
        FIXED_ANCHOR_METHOD_ID
    )
    assert tuple(document["hand_binding"]["prepared_pad_order"]) == PAD_NAMES
    assert [row["anchor_pad_ordinal"] for row in document["lanes"]] == [
        None,
        0,
        1,
        2,
    ]
    assert all(
        len(row["maximum_prefix_design_sha256"]) == 64
        for row in document["lanes"]
    )
    assert document["external_lane_registry_supported"] is False
    assert document["local_refinement"] == {
        "method_id": "CANONICAL_V9_DYADIC_STENCIL_V1",
        "execution_status": "DISABLED_FOR_V1",
        "evaluation_budget": 0,
        "ranking_eligible": False,
    }
    assert [row["lane"] for row in document["lanes"]] == [
        "DIRECT_V9",
        "SURFACE_PAD_A",
        "SURFACE_PAD_B",
        "SURFACE_PAD_C",
    ]
    assert current.contract_hash_sha256 != _generator(
        pad_names=("other_a", "other_b", "other_c")
    ).contract_hash_sha256
    assert current.contract_hash_sha256 != _generator(
        v9=_AcceptingV9(model_token="alternative_model")
    ).contract_hash_sha256

    class _AlternativeProjectingSurface(_ProjectingSurface):
        pass

    assert current.contract_hash_sha256 != _generator(
        surface=_AlternativeProjectingSurface(),
        hand_model={"object": "current"},
    ).contract_hash_sha256


def test_constructor_rejects_every_runtime_provenance_mismatch() -> None:
    def components() -> tuple[object, _AcceptingV9, _ProjectingSurface]:
        hand = object()
        v9 = _AcceptingV9()
        v9.bind_contract(hand, PAD_NAMES)
        surface = _ProjectingSurface()
        surface.bind_contract(v9, hand, PAD_NAMES)
        return hand, v9, surface

    hand, v9, surface = components()
    v9.method_id = "NOT_PRODUCTION_V9"
    with pytest.raises(TopLevelCandidateGeneratorError, match="method_id"):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    v9.parameter_layout = V9_PARAMETER_LAYOUT_PREFIX + (
        "preshape_joint_unit:f2j1",
    )
    with pytest.raises(
        TopLevelCandidateGeneratorError, match="parameter_layout"
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    v9.model_binding_complete = False
    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="model_binding_complete",
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    v9.model_binding_status = MODEL_BINDING_UNBOUND_STATUS
    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="model_binding_status",
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    del v9.model_contract_sha256
    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="model_contract_sha256",
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    v9.model_contract_canonical_json += " "
    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="not canonical",
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    surface.closure_model = _AcceptingV9()
    with pytest.raises(
        TopLevelCandidateGeneratorError, match="exact V9 evaluator"
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    surface.hand_model = object()
    with pytest.raises(
        TopLevelCandidateGeneratorError, match="exact supplied hand"
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )

    hand, v9, surface = components()
    surface.prepared_pad_names = ("pad_b", "pad_a", "pad_c")
    with pytest.raises(
        TopLevelCandidateGeneratorError, match="prepared_pad_names"
    ):
        TopLevelCandidateGenerator(
            v9_evaluator=v9,
            surface_proposer=surface,
            hand_model=hand,
            anchor_pad_names=PAD_NAMES,
        )


def test_strict_constructor_and_budget_reject_implicit_extensions() -> None:
    with pytest.raises(
        TopLevelCandidateGeneratorError, match="three distinct"
    ):
        _generator(pad_names=("pad_a", "pad_a", "pad_c"))
    with pytest.raises(TopLevelCandidateGeneratorError, match="hand_model"):
        TopLevelCandidateGenerator(
            v9_evaluator=_AcceptingV9(),
            surface_proposer=_ProjectingSurface(),
            hand_model=None,
            anchor_pad_names=PAD_NAMES,
        )

    generator = _generator()
    for invalid_budget in (64, 129, 1024, True, 256.0):
        with pytest.raises(
            TopLevelCandidateGeneratorError,
            match="exactly one of 128, 256 or 512",
        ):
            generator.generate(invalid_budget)  # type: ignore[arg-type]


def test_default_result_is_main_budget_static_only_and_unranked() -> None:
    v9 = _AcceptingV9()
    result = _generator(v9=v9).generate()

    assert result.total_attempt_budget == MAIN_TOTAL_ATTEMPT_BUDGET
    assert result.local_refinement_evaluation_budget == 0
    assert result.v9_evaluation_count == len(result.unique_v9_evaluations)
    assert len(v9.calls) == result.v9_evaluation_count
    assert all(row.accepted_static for row in result.unique_v9_evaluations)
    assert all(
        row.invocation_binding is not None
        and row.invocation_binding.requested_parameters_unit
        == row.v9_parameters_unit
        and row.invocation_binding.requested_parameter_key_hex
        == row.v9_parameter_key_hex
        for row in result.unique_v9_evaluations
    )
    assert len(result.accepted_candidates) == result.v9_evaluation_count
    assert not hasattr(result, "ranked")
    assert not hasattr(result, "selected")


def test_contact_range_policy_has_a_separate_top_level_output_channel() -> None:
    v9 = _PolicyV9()
    result = _generator(v9=v9).generate(128)

    assert not result.accepted_candidates
    assert len(result.accepted_policies) == result.v9_evaluation_count
    assert all(
        row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
        and row.candidate is None
        and type(row.sequential_closure_policy)
        is CertifiedSequentialClosurePolicy
        and row.v9_failure_reason is None
        for row in result.unique_v9_evaluations
    )
    assert all(
        row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
        and row.failure_reason is None
        for row in result.attempts
    )
    assert all(
        row.sequential_closure_policy.policy_sha256
        == result.unique_v9_evaluations[index].sequential_closure_policy.policy_sha256
        for index, row in enumerate(result.accepted_policies)
    )


def test_contact_range_policy_checkpoint_is_exact_and_rejects_binding_drift() -> None:
    generator = _generator(v9=_PolicyV9())
    state = generator.advance_resumable(
        generator.begin_resumable(128),
        stop_attempt_index_exclusive=8,
    )
    codec = _checkpoint_codec()
    encoded = codec.canonical_bytes(state)
    decoded = codec.decode_canonical_bytes(encoded)

    assert codec.canonical_bytes(decoded) == encoded
    assert b"DisplayOnlyGraspProposal" not in encoded
    generator.validate_resumable_state(decoded)
    first = decoded.unique_v9_evaluations[0]
    policy = first.sequential_closure_policy
    assert type(policy) is CertifiedSequentialClosurePolicy
    changed_policy = replace(
        policy,
        model_contract_sha256="0" * 64,
    )
    changed_state = replace(
        decoded,
        unique_v9_evaluations=(
            replace(
                first,
                sequential_closure_policy=changed_policy,
            ),
            *decoded.unique_v9_evaluations[1:],
        ),
    )
    changed_roundtrip = codec.decode_canonical_bytes(
        codec.canonical_bytes(changed_state)
    )
    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="stored V9 policy differs",
    ):
        generator.validate_resumable_state(changed_roundtrip)


def _checkpoint_codec() -> CanonicalCheckpointCodec:
    return CanonicalCheckpointCodec(additional_allowed_types=(_Audit,))


def _canonical_result_bytes(value: object) -> bytes:
    return _checkpoint_codec().canonical_bytes(value)


@pytest.mark.parametrize("budget", ALLOWED_TOTAL_ATTEMPT_BUDGETS)
def test_resumable_segments_are_byte_exactly_one_shot(
    budget: int,
) -> None:
    expected = _generator().generate(budget)
    generator = _generator()
    state = generator.begin_resumable(budget)
    boundaries = sorted(
        {
            1,
            min(7, budget),
            budget // 3,
            budget // 2,
            budget - 1,
            budget,
        }
    )
    for boundary in boundaries:
        state = generator.advance_resumable(
            state,
            stop_attempt_index_exclusive=boundary,
        )
    observed = generator.finalize_prefix(state, budget)

    assert observed == expected
    assert _canonical_result_bytes(observed) == (
        _canonical_result_bytes(expected)
    )
    with pytest.raises(FrozenInstanceError):
        state.target_total_attempt_budget = 512


def test_resumable_target_extension_preserves_every_frozen_prefix() -> None:
    generator = _generator()
    state = generator.begin_resumable(128)
    expected_by_budget = {
        budget: _generator().generate(budget)
        for budget in ALLOWED_TOTAL_ATTEMPT_BUDGETS
    }

    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=128,
    )
    assert _canonical_result_bytes(
        generator.finalize_prefix(state, 128)
    ) == _canonical_result_bytes(expected_by_budget[128])
    state = generator.extend_target(state, 256)
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=193,
    )
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=256,
    )
    assert _canonical_result_bytes(
        generator.finalize_prefix(state, 256)
    ) == _canonical_result_bytes(expected_by_budget[256])
    state = generator.extend_target(state, 512)
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=512,
    )

    for budget in ALLOWED_TOTAL_ATTEMPT_BUDGETS:
        assert _canonical_result_bytes(
            generator.finalize_prefix(state, budget)
        ) == _canonical_result_bytes(expected_by_budget[budget])


def test_duplicate_and_exception_outcomes_survive_segment_boundaries() -> None:
    v9 = _RejectZeroV9()
    generator = _generator(
        v9=v9,
        surface=_ConstantSurface((1.0, -0.0, 0.0, -0.0, 0.0)),
    )
    state = generator.begin_resumable(128)
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=2,
    )
    calls_before_duplicate = tuple(v9.calls)
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=3,
    )
    assert tuple(v9.calls) == calls_before_duplicate
    state = generator.advance_resumable(
        state,
        stop_attempt_index_exclusive=128,
    )
    expected = _generator(
        v9=_RejectZeroV9(),
        surface=_ConstantSurface((1.0, -0.0, 0.0, -0.0, 0.0)),
    ).generate(128)
    assert _canonical_result_bytes(
        generator.finalize_prefix(state, 128)
    ) == _canonical_result_bytes(expected)
    assert len(v9.calls) == len(set(v9.calls))

    raising_generator = _generator(v9=_RaisingV9())
    raising_state = raising_generator.begin_resumable(128)
    for stop in (1, 31, 64, 128):
        raising_state = raising_generator.advance_resumable(
            raising_state,
            stop_attempt_index_exclusive=stop,
        )
    raising_result = raising_generator.finalize_prefix(raising_state, 128)
    expected_raising = _generator(v9=_RaisingV9()).generate(128)
    assert _canonical_result_bytes(raising_result) == (
        _canonical_result_bytes(expected_raising)
    )
    assert all(
        row.status is AttemptStatus.V9_EVALUATOR_EXCEPTION
        for row in raising_result.unique_v9_evaluations
    )


def test_resumable_state_tamper_fails_closed() -> None:
    generator = _generator()
    state = generator.advance_resumable(
        generator.begin_resumable(128),
        stop_attempt_index_exclusive=1,
    )
    first = state.attempts[0]
    sobol = list(first.lineage.sobol_parameters_unit)
    sobol[0] = math.nextafter(sobol[0], 1.0)
    tampered_attempt = replace(
        first,
        lineage=replace(
            first.lineage,
            sobol_parameters_unit=tuple(sobol),
        ),
    )
    tampered = replace(
        state,
        attempts=(tampered_attempt,),
    )

    with pytest.raises(
        TopLevelCandidateGeneratorError,
        match="schedule/Sobol lineage changed",
    ):
        generator.validate_resumable_state(tampered)


def test_checkpoint_codec_is_binary64_exact_and_explicitly_whitelisted(
) -> None:
    codec = _checkpoint_codec()
    encoded = codec.canonical_bytes((-0.0, 0.0, math.nextafter(1.0, 0.0)))
    decoded = codec.decode_canonical_bytes(encoded)

    assert _binary64_bytes(decoded) == _binary64_bytes(
        (-0.0, 0.0, math.nextafter(1.0, 0.0))
    )
    with pytest.raises(GenerationCheckpointError, match="not explicitly"):
        codec.canonical_bytes(SimpleNamespace(value=1))
    with pytest.raises(GenerationCheckpointError, match="duplicate key"):
        codec.decode_canonical_bytes(b'{"$tuple":[],"$tuple":[]}')
    with pytest.raises(
        GenerationCheckpointError,
        match="cannot contain decimal floats",
    ):
        codec.decode_canonical_bytes(b'{"$tuple":[0.5]}')


def _binary64_bytes(value: object) -> bytes:
    return np.asarray(value, dtype=">f8").tobytes(order="C")


def _checkpoint_environment_sha256() -> str:
    return hashlib.sha256(b"checkpoint test environment").hexdigest()


def test_checkpoint_store_typed_restore_and_128_to_256_extension(
    tmp_path: object,
) -> None:
    root = tmp_path / "generation_checkpoint"
    environment = _checkpoint_environment_sha256()
    run_id = "test-typed-cross-process-restore"
    codec = _checkpoint_codec()
    first_generator = _generator()
    first_store = GenerationCheckpointStore(root, codec=codec)
    stored = first_store.initialize(
        generator=first_generator,
        state=first_generator.begin_resumable(128),
        run_id=run_id,
        execution_environment_sha256=environment,
    )
    intent = first_store.commit_intent(
        stored,
        generator=first_generator,
        stop_attempt_index_exclusive=17,
    )
    advanced = first_generator.advance_resumable(
        intent.state,
        stop_attempt_index_exclusive=17,
    )
    first_store.commit_advanced(
        intent,
        generator=first_generator,
        advanced_state=advanced,
    )

    second_generator = _generator()
    second_store = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    restored = second_store.load_latest(
        generator=second_generator,
        run_id=run_id,
        execution_environment_sha256=environment,
    )
    assert restored.state.completed_attempt_count == 17
    assert _checkpoint_codec().canonical_bytes(restored.state) == (
        _checkpoint_codec().canonical_bytes(advanced)
    )
    intent = second_store.commit_intent(
        restored,
        generator=second_generator,
        stop_attempt_index_exclusive=128,
    )
    advanced = second_generator.advance_resumable(
        intent.state,
        stop_attempt_index_exclusive=128,
    )
    completed = second_store.commit_advanced(
        intent,
        generator=second_generator,
        advanced_state=advanced,
    )
    assert completed.manifest.lifecycle is (
        CheckpointLifecycle.PREFIX_COMPLETE
    )

    extended_state = second_generator.extend_target(completed.state, 256)
    extended = second_store.commit_extension(
        completed,
        generator=second_generator,
        extended_state=extended_state,
    )
    assert extended.manifest.lifecycle is CheckpointLifecycle.READY
    third_generator = _generator()
    third_store = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    restored_extension = third_store.load_latest(
        generator=third_generator,
        run_id=run_id,
        execution_environment_sha256=environment,
    )
    intent = third_store.commit_intent(
        restored_extension,
        generator=third_generator,
        stop_attempt_index_exclusive=256,
    )
    final_state = third_generator.advance_resumable(
        intent.state,
        stop_attempt_index_exclusive=256,
    )
    final_checkpoint = third_store.commit_advanced(
        intent,
        generator=third_generator,
        advanced_state=final_state,
    )
    observed = third_generator.finalize_prefix(
        final_checkpoint.state,
        256,
    )
    expected = _generator().generate(256)
    assert _canonical_result_bytes(observed) == (
        _canonical_result_bytes(expected)
    )

    state_512 = third_generator.extend_target(
        final_checkpoint.state,
        512,
    )
    checkpoint_512 = third_store.commit_extension(
        final_checkpoint,
        generator=third_generator,
        extended_state=state_512,
    )
    fourth_generator = _generator()
    fourth_store = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    restored_512 = fourth_store.load_latest(
        generator=fourth_generator,
        run_id=run_id,
        execution_environment_sha256=environment,
    )
    intent_512 = fourth_store.commit_intent(
        restored_512,
        generator=fourth_generator,
        stop_attempt_index_exclusive=512,
    )
    advanced_512 = fourth_generator.advance_resumable(
        intent_512.state,
        stop_attempt_index_exclusive=512,
    )
    completed_512 = fourth_store.commit_advanced(
        intent_512,
        generator=fourth_generator,
        advanced_state=advanced_512,
    )
    expected_512 = _generator().generate(512)
    observed_512 = fourth_generator.finalize_prefix(
        completed_512.state,
        512,
    )
    assert _canonical_result_bytes(observed_512) == (
        _canonical_result_bytes(expected_512)
    )
    all_requests = tuple(
        first_generator.v9_evaluator.calls
        + second_generator.v9_evaluator.calls
        + third_generator.v9_evaluator.calls
        + fourth_generator.v9_evaluator.calls
    )
    assert len(all_requests) == len(set(all_requests))
    assert checkpoint_512.state.target_total_attempt_budget == 512


def test_in_flight_checkpoint_recovery_is_blocked_without_retry(
    tmp_path: object,
) -> None:
    root = tmp_path / "in_flight"
    generator = _generator()
    store = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    environment = _checkpoint_environment_sha256()
    stored = store.initialize(
        generator=generator,
        state=generator.begin_resumable(128),
        run_id="in-flight-test",
        execution_environment_sha256=environment,
    )
    intent = store.commit_intent(
        stored,
        generator=generator,
        stop_attempt_index_exclusive=1,
    )
    ambiguous_advanced_state = generator.advance_resumable(
        intent.state,
        stop_attempt_index_exclusive=1,
    )
    calls_after_ambiguous_evaluation = tuple(
        generator.v9_evaluator.calls
    )

    restarted = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    inspected = restarted.inspect_latest(
        generator=generator,
        run_id="in-flight-test",
        execution_environment_sha256=environment,
    )
    assert inspected.manifest.lifecycle is CheckpointLifecycle.IN_FLIGHT
    with pytest.raises(
        GenerationCheckpointRecoveryBlocked,
        match="RECOVERY_BLOCKED_IN_FLIGHT",
    ):
        restarted.load_latest(
            generator=generator,
            run_id="in-flight-test",
            execution_environment_sha256=environment,
        )
    with pytest.raises(
        GenerationCheckpointRecoveryBlocked,
        match="this process did not commit the evaluator intent",
    ):
        restarted.commit_advanced(
            inspected,
            generator=generator,
            advanced_state=ambiguous_advanced_state,
        )
    assert tuple(generator.v9_evaluator.calls) == (
        calls_after_ambiguous_evaluation
    )


@pytest.mark.parametrize(
    ("fault_stage", "latest_is_blocked"),
    (
        ("after_state_blob_write", False),
        ("after_manifest_write", False),
        ("before_latest_replace", False),
        ("after_latest_replace", True),
    ),
)
def test_atomic_checkpoint_faults_leave_old_or_blocked_latest(
    tmp_path: object,
    fault_stage: str,
    latest_is_blocked: bool,
) -> None:
    root = tmp_path / fault_stage
    generator = _generator()
    codec = _checkpoint_codec()
    environment = _checkpoint_environment_sha256()
    base_store = GenerationCheckpointStore(root, codec=codec)
    base = base_store.initialize(
        generator=generator,
        state=generator.begin_resumable(128),
        run_id="atomic-fault-test",
        execution_environment_sha256=environment,
    )

    def inject(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"injected:{stage}")

    failing_store = GenerationCheckpointStore(
        root,
        codec=_checkpoint_codec(),
        fault_injector=inject,
    )
    with pytest.raises(RuntimeError, match="injected"):
        failing_store.commit_intent(
            base,
            generator=generator,
            stop_attempt_index_exclusive=1,
        )

    reader = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    if latest_is_blocked:
        with pytest.raises(GenerationCheckpointRecoveryBlocked):
            reader.load_latest(
                generator=generator,
                run_id="atomic-fault-test",
                execution_environment_sha256=environment,
            )
    else:
        restored = reader.load_latest(
            generator=generator,
            run_id="atomic-fault-test",
            execution_environment_sha256=environment,
        )
        assert restored.checkpoint_sha256 == base.checkpoint_sha256


def test_checkpoint_blob_tamper_and_stale_writer_fail_closed(
    tmp_path: object,
) -> None:
    root = tmp_path / "tamper"
    generator = _generator()
    environment = _checkpoint_environment_sha256()
    store = GenerationCheckpointStore(root, codec=_checkpoint_codec())
    base = store.initialize(
        generator=generator,
        state=generator.begin_resumable(128),
        run_id="tamper-test",
        execution_environment_sha256=environment,
    )
    forged_base = replace(
        base,
        state=generator.extend_target(base.state, 256),
    )
    with pytest.raises(
        GenerationCheckpointError,
        match="differs from authoritative CAS bytes",
    ):
        store.commit_intent(
            forged_base,
            generator=generator,
            stop_attempt_index_exclusive=1,
        )

    intent = store.commit_intent(
        base,
        generator=generator,
        stop_attempt_index_exclusive=1,
    )
    with pytest.raises(
        GenerationCheckpointError,
        match="concurrent writer",
    ):
        store.commit_intent(
            base,
            generator=generator,
            stop_attempt_index_exclusive=1,
        )

    blob_path = (
        store.blob_directory
        / f"{intent.manifest.state_blob_sha256}.json"
    )
    original = blob_path.read_bytes()
    blob_path.write_bytes(original[:-1] + b" ")
    with pytest.raises(
        GenerationCheckpointError,
        match="contradict digest",
    ):
        store.inspect_latest(
            generator=generator,
            run_id="tamper-test",
            execution_environment_sha256=environment,
        )
