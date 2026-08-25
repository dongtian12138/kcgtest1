"""Direct tests for bounded height projection before the Top-8 budget."""

from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.height_projected_search import (
    SampledPathEnvelope,
    search_height_projected_pregrasps,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    ClosurePrediction,
    FastFilterResult,
)


class _Hand:
    independent_joint_names = ("f1j1", "f1j2", "f2j1", "f3j2")
    joints = {name: SimpleNamespace(limit=SimpleNamespace(lower=0.0, upper=1.57))
              for name in independent_joint_names}

    def resolve_joint_positions(self, positions, *, enforce_limits):
        assert enforce_limits
        return dict(zip(self.independent_joint_names, positions))

    def forward_kinematics(self, joints, *, base_transform):
        assert len(joints) == 4
        return {"link": np.asarray(base_transform)}


class _Config:
    def section(self, name):
        return {
            "candidate_generation": {
                "backend": "GRASPGENX_FULL_PALM", "maximum_closure_phase": 0.75},
            "closure_prediction": {
                "closing_order": ("pad_1", "pad_2", "pad_3"),
                "contact_distance_m": 0.05, "phase_sample_count": 3,
                "phase_sampling_rule": "DYNAMIC_CONTROL_STEP_BOUNDED"},
            "dynamic": {"finger_maximum_speed_rad_s": 12.0,
                        "physics_dt_s": 1.0 / 120.0},
        }[name]


INPUTS = SimpleNamespace(
    hand_model=_Hand(), config=_Config(),
    closing_directions=np.asarray(((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))),
    frozen_world_from_object=np.eye(4),
    table_xy_bounds_m=np.asarray(((-1.0, 1.0), (-1.0, 1.0))),
    table_top_z_m=0.0,
    hand_collision_triangles_by_link={"link": np.asarray([
        [[-0.01, -0.01, -0.10], [0.01, -0.01, -0.10], [0.0, 0.01, -0.10]]])},
    hand_contract=SimpleNamespace(pads=tuple(
        SimpleNamespace(name=name) for name in ("pad_1", "pad_2", "pad_3"))),
    task_grip_surfaces={name: SimpleNamespace(
        link_name="link", points_local_m=np.asarray((
            (-0.01, -0.01, 0.0), (0.01, -0.01, 0.0), (0.0, 0.01, 0.0))))
        for name in ("pad_1", "pad_2", "pad_3")},
    face_roles=SimpleNamespace(face_is_allowed=np.asarray((True,))),
    object_contract=SimpleNamespace(model=SimpleNamespace(mesh=SimpleNamespace(
        face_vertices_m=np.asarray([[
            [-0.01, -0.01, 0.2], [0.01, -0.01, 0.2], [0.0, 0.01, 0.2]]])))),
)


def _seed() -> CandidateSeed:
    return CandidateSeed(
        candidate_id="seed", object_id="object", anchor_face_index=0,
        anchor_position_object_m=(0.0, 0.0, 0.0),
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        pregrasp_joint_positions_rad=(0.7, 0.0, 0.0, 0.0),
        pregrasp_closure_phases=(0.0, 0.0, 0.0), source_sample_index=0,
        palm_configuration_rad=0.7,
    )


def _prediction(seed, passed):
    contacts = tuple(SimpleNamespace(
        pad_name=f"pad_{index}", object_face_index=index,
        hand_surface_face_index=10 + index, hand_surface_legacy_blue_pad=False,
    ) for index in range(1, 4))
    return ClosurePrediction(
        seed=seed, status="CLOSURE_SURVIVE" if passed else "CLOSURE_REJECT",
        contacts=contacts if passed else (), final_joint_positions_rad=(0.7, 0, 0, 0),
        final_closure_phases=(0.5, 0.5, 0.5),
        minimum_initial_pad_clearance_m=0.01,
        reason="" if passed else "NO_THREE_CONTACT",
    )


class _Predictor:
    def __init__(self, interval=(0.15, 0.25)):
        self.inputs = INPUTS
        self.interval = interval

    def predict(self, seed):
        height = seed.object_from_hand_matrix()[2, 3]
        return _prediction(seed, self.interval[0] <= height <= self.interval[1])


def _envelope(seed):
    base = INPUTS.frozen_world_from_object @ seed.object_from_hand_matrix()
    joints = np.asarray(seed.pregrasp_joint_positions_rad)
    stages = (
        "PALM_FAR_0000", "PRESHAPE_FAR_0001", "APPROACH_00", "PREGRASP",
        "FINGER_1_CLOSURE_0001", "FINGER_2_CLOSURE_0001",
        "FINGER_3_CLOSURE_0001", "PRELOAD_END", "LIFT_START",
    )
    return SampledPathEnvelope(
        tuple((stage, base, joints) for stage in stages),
        "REGISTERED_COMPLETE_CONTROL_STEP_PATH",
    )


def _fast(prediction, status="FAST_SURVIVE", sweep=True):
    return FastFilterResult(
        candidate_id=prediction.seed.candidate_id, status=status,
        reasons=() if status == "FAST_SURVIVE" else ("TABLE_COLLISION",),
        unresolved_checks=(), sequential_closure_sweep_pass=sweep,
        checked_state_count=9, minimum_table_clearance_m=0.05,
    )


def _search(predictor, fast, **overrides):
    arguments = dict(
        sampled_path_envelope=_envelope,
        pregrasp_contact_key=lambda bound: (
            sum(abs(value - 0.1) for value in bound.pregrasp_closure_phases),),
        pregrasp_path_callback=lambda bound: {
            "candidate_id": bound.candidate_id,
            "pregrasp_closure_phases": bound.pregrasp_closure_phases,
            "accepted": True, "reasons": (),
            "minimum_table_clearance_m": 0.05,
        },
        fast_filter_callback=fast,
        contact_height_bounds_m=(0.0, 0.4),
        coarse_sample_count=9,
        boundary_tolerance_m=1.0e-6,
        maximum_bisection_iterations=32,
        table_numerical_tolerance_m=1.0e-5,
        required_table_clearance_m=0.001,
    )
    arguments.update(overrides)
    return search_height_projected_pregrasps(
        INPUTS, _seed(), predictor, **arguments)


def test_all_27_are_ranked_before_one_bounded_variant_is_projected() -> None:
    contact_calls, fast_calls = [], []

    def contact_key(bound):
        contact_calls.append(bound.pregrasp_closure_phases)
        return (sum(abs(value - 0.1) for value in bound.pregrasp_closure_phases),)

    survivors, audit = _search(
        _Predictor(), lambda prediction: fast_calls.append(prediction) or _fast(prediction),
        pregrasp_contact_key=contact_key,
    )
    assert len(contact_calls) == 27
    assert len(audit["deferred"]) == 26
    assert len(survivors) == audit["exact_variant_evaluated_count"] == 1
    assert len(fast_calls) == 1
    assert survivors[0].object_from_hand_matrix()[2, 3] == pytest.approx(0.15, abs=2e-6)
    row = audit["evaluated"][0]
    assert row["minimum_table_handbase_z_m"] == pytest.approx(0.101)
    assert row["fresh_checked_state_count"] == 9


def test_empty_contact_table_intersection_is_hard_reject() -> None:
    survivors, audit = _search(_Predictor(), _fast,
                               maximum_exact_variants=1,
                               required_table_clearance_m=0.4)
    assert survivors == ()
    assert audit["evaluated"][0]["reason"] == (
        "EMPTY_TABLE_AND_CONTACT_HEIGHT_INTERSECTION_CONSERVATIVE_GATE")


def test_projected_candidate_requires_fresh_complete_sequential_sweep() -> None:
    survivors, audit = _search(
        _Predictor(), lambda prediction: _fast(prediction, sweep=False),
        maximum_exact_variants=1,
    )
    assert survivors == ()
    row = audit["evaluated"][0]
    assert row["fresh_closure_status"] == "CLOSURE_SURVIVE"
    assert row["fresh_sequential_closure_sweep_pass"] is False
    assert row["status"] == "POST_PROJECTION_REVALIDATION_REJECT"
    assert row["reason"] == "PROJECTED_HEIGHT_REVALIDATION_FAILED"


def test_incomplete_path_envelope_fails_closed() -> None:
    base = np.eye(4)
    incomplete = lambda seed: SampledPathEnvelope((
        ("PREGRASP", base, np.asarray(seed.pregrasp_joint_positions_rad)),
    ), "INCOMPLETE")
    with pytest.raises(ValueError, match="complete registered path"):
        _search(_Predictor(), _fast, sampled_path_envelope=incomplete,
                maximum_exact_variants=1)
