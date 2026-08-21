from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

from kcg_connector.grasp.robust.grasp_optimizer import deterministic_sobol
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.post_generation_ranker import (
    COMPLETE_CLEARANCE_SCOPE,
    METHOD_ID as POST_GENERATION_RANKER_METHOD_ID,
    PostGenerationRankOnlyPipeline,
    SCENARIO_COUNT as POST_GENERATION_SCENARIO_COUNT,
    SCENARIO_DESIGN_SHA256 as POST_GENERATION_SCENARIO_DESIGN_SHA256,
    SCENARIO_DIMENSION as POST_GENERATION_SCENARIO_DIMENSION,
    SCENARIO_METHOD_ID as POST_GENERATION_SCENARIO_METHOD_ID,
    SCENARIO_SOBOL_SEED as POST_GENERATION_SCENARIO_SOBOL_SEED,
)
from kcg_connector.grasp.robust.study_contract import (
    StudyContractError,
    StudyFreezeIncompleteError,
    _validate_ray_closure_hand_binding,
    audit_study_contract,
    build_frozen_study_contract,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
)


REPOSITORY = Path(__file__).resolve().parents[4]
METHOD = REPOSITORY / "src/kcg_connector/config/carts_grasp_v1.yaml"
HAND = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
OBJECTS = REPOSITORY / "src/kcg_connector/config/carts_grasp_objects_v1.yaml"
DEVELOPMENT = "current_d38999_26kj61sn_public_spec"
TRANSFER = "te_deutsch_d38999_26fj35pn_step"


def _audit(method: Path = METHOD, hand: Path = HAND, objects: Path = OBJECTS):
    return audit_study_contract(
        method,
        hand,
        objects,
        repository_root=REPOSITORY,
    )


def _document(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _reverse_mapping_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(child)
            for key, child in reversed(tuple(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(child) for child in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_contract_audit_binds_top_generator_but_retains_true_blockers(
) -> None:
    audit = _audit()

    assert audit.freeze_eligible is False
    assert audit.preregistration_blockers == (
        "MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION",
        "MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING",
        "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS",
    )
    assert audit.canonical_sha256 == hashlib.sha256(
        audit.canonical_json_bytes
    ).hexdigest()
    assert audit.canonical_sha256 == (
        "40e019e06a30970568c7ed1150c59c3bbf7c7a30a38446ac3ca1460d6a687c43"
    )
    assert audit.canonical_manifest["status"] == "NOT_FREEZE_ELIGIBLE"
    assert audit.canonical_manifest["study"]["transfer_object_role"] == (
        "ZERO_OBJECT_TUNING_HELD_OUT_CASE_STUDY"
    )
    assert audit.canonical_manifest["claim_boundaries"][
        "transfer_geometry_seen_before_method_freeze"
    ] is True
    assert audit.canonical_manifest["claim_boundaries"][
        "prospective_double_blind_claim_allowed"
    ] is False
    assert audit.canonical_manifest["protocol_interpretation"][
        "numerical_tolerances"
    ] == "NUMERICAL_SOLVER_PROTOCOL_NOT_PHYSICAL_ACCEPTANCE_GATE"
    assert audit.canonical_manifest["protocol_interpretation"][
        "ray_closure_subdivision_budget"
    ] == "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
    assert audit.canonical_manifest["protocol_interpretation"][
        "ray_closure_budget_convergence"
    ] == "COMPUTE_CONVERGENCE_AUDIT_NOT_PHYSICAL_PASS_GATE"
    assert audit.canonical_manifest["protocol_interpretation"][
        "ray_closure_adjacent_budget_result_stability_report_required"
    ] is True
    assert audit.canonical_manifest["protocol_interpretation"][
        "ray_closure_physical_acceptance_gate"
    ] is False
    assert audit.canonical_manifest["protocol_interpretation"][
        "interval_backend_decimal_precision"
    ] == 80
    assert audit.canonical_manifest["protocol_interpretation"][
        "interval_backend_root_bisection_budget"
    ] == "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
    assert audit.canonical_manifest["protocol_interpretation"][
        "interval_backend_physical_acceptance_gate"
    ] is False
    interpretation = audit.canonical_manifest["protocol_interpretation"]
    assert interpretation["top_level_candidate_generator_method_id"] == (
        "CARTS_DIRECT_V9_PLUS_FIXED_SINGLE_ANCHOR_STRATIFIED_GENERATOR_V1"
    )
    assert interpretation[
        "top_level_candidate_main_total_attempt_budget"
    ] == 256
    assert tuple(
        interpretation["top_level_candidate_allowed_total_attempt_budgets"]
    ) == (128, 256, 512)
    assert interpretation["top_level_candidate_no_replacement"] is True
    assert interpretation[
        "top_level_candidate_failure_and_duplicate_consume_attempt"
    ] is True
    assert interpretation["top_level_candidate_scipy_version"] == "1.8.0"
    assert interpretation[
        "top_level_candidate_realized_design_hashes_bound"
    ] is True
    assert interpretation[
        "top_level_candidate_local_refinement_execution_status"
    ] == "DISABLED_FOR_V1"
    assert interpretation["top_level_candidate_output_claim"] == (
        "STATIC_V9_EXACT_CANDIDATE_OR_CONTACT_RANGE_POLICY_"
        "ACCEPTANCE_ONLY"
    )
    assert interpretation[
        "top_level_candidate_accepted_output_channels"
    ] == [
        "STATIC_EXACT_GRASP_CANDIDATE",
        "STATIC_CERTIFIED_SEQUENTIAL_CLOSURE_POLICY",
    ]
    assert interpretation[
        "top_level_candidate_and_policy_mutually_exclusive"
    ] is True
    assert interpretation[
        "top_level_display_only_proposal_formal_eligible"
    ] is False
    assert interpretation[
        "top_level_contact_range_policy_downstream_status"
    ] == "PENDING_POLICY_AWARE_COLLISION_AND_WRENCH"
    assert interpretation["legacy_grasp_optimizer_formal_eligible"] is False
    assert interpretation["post_generation_ranking_binding_status"] == (
        "BOUND_TO_PRODUCTION_POST_GENERATION_RANK_ONLY_PIPELINE"
    )
    assert interpretation[
        "post_generation_ranking_implementation_type_id"
    ] == (
        f"{PostGenerationRankOnlyPipeline.__module__}."
        f"{PostGenerationRankOnlyPipeline.__qualname__}"
    )
    assert interpretation["post_generation_ranking_method_id"] == (
        POST_GENERATION_RANKER_METHOD_ID
    )
    assert interpretation[
        "post_generation_ranking_common_scenario_method_id"
    ] == POST_GENERATION_SCENARIO_METHOD_ID
    assert interpretation[
        "post_generation_policy_aware_ranking_guard"
    ] == "POLICY_AWARE_COLLISION_AND_WRENCH_REQUIRED_BEFORE_RANKING"
    assert interpretation[
        "post_generation_policy_collision_invocations_before_support"
    ] == 0
    assert interpretation[
        "post_generation_policy_wrench_invocations_before_support"
    ] == 0
    assert interpretation[
        "post_generation_ranking_common_scenario_dimension"
    ] == POST_GENERATION_SCENARIO_DIMENSION
    assert interpretation[
        "post_generation_ranking_common_scenario_count"
    ] == POST_GENERATION_SCENARIO_COUNT
    assert interpretation[
        "post_generation_ranking_common_scenario_sobol_seed"
    ] == POST_GENERATION_SCENARIO_SOBOL_SEED
    assert interpretation[
        "post_generation_ranking_common_scenario_design_sha256"
    ] == POST_GENERATION_SCENARIO_DESIGN_SHA256
    assert interpretation[
        "post_generation_ranking_formal_selection_allowed"
    ] is False
    assert interpretation[
        "post_generation_ranking_required_collision_scope"
    ] == COMPLETE_CLEARANCE_SCOPE
    assert interpretation["uncertainty_nested_convergence_role"] == (
        "SAME_SEED_SCRAMBLED_SOBOL_PREFIX_EXTENSION_"
        "NOT_INDEPENDENT_VALIDATION"
    )
    assert dict(audit.canonical_manifest["dynamic_eligibility"]) == {
        DEVELOPMENT: False,
        TRANSFER: False,
    }
    ray_closure = audit.canonical_manifest["source_documents"][
        "shared_method"
    ]["ray_closure"]
    assert ray_closure["maximum_subdivision_intervals"] == 4096
    assert tuple(ray_closure["budget_convergence_values"]) == (
        1024,
        2048,
        4096,
        8192,
    )
    domain = ray_closure["closure_parameter_domain"]
    assert domain["dimension"] == 5
    assert tuple(domain["parameter_layout"]) == (
        "assembly_axis_yaw_unit",
        "axial_target_unit",
        "lateral_task_x_unit",
        "lateral_task_y_unit",
        "preshape_joint_unit:f1j1",
    )
    assert domain["assembly_axis_yaw"]["unit_interval"] == (
        "HALF_OPEN_ZERO_TO_ONE"
    )
    assert domain["placement"]["unit_interval"] == "CLOSED_ZERO_TO_ONE"
    assert domain["preshape"]["unit_interval"] == "CLOSED_ZERO_TO_ONE"
    interval_backend = ray_closure["interval_backend"]
    assert interval_backend["method_id"] == (
        "MPMATH_DIRECTED_INTERVAL_SECOND_ORDER_URDF_JET_V1"
    )
    assert interval_backend["decimal_precision"] == 80
    assert tuple(interval_backend["precision_convergence_values"]) == (
        50,
        80,
        120,
    )
    assert interval_backend["maximum_root_bisection_iterations"] == 256
    assert tuple(interval_backend["root_budget_convergence_values"]) == (
        128,
        256,
        512,
    )
    method = audit.canonical_manifest["source_documents"]["shared_method"]
    candidate = method["candidate_optimization"]
    assert candidate["method_id"] == (
        "CARTS_DIRECT_V9_PLUS_FIXED_SINGLE_ANCHOR_STRATIFIED_GENERATOR_V1"
    )
    assert candidate["v9_certifier"]["parameter_layout"][-1] == (
        "preshape_joint_unit:f1j1"
    )
    assert candidate["fixed_anchor_mapper"]["parameter_layout"][-1] == (
        "preshape_joint_unit:f1j1"
    )
    assert tuple(candidate["hand_binding"]["prepared_pad_order"]) == (
        "finger_1_pad",
        "finger_2_pad",
        "finger_3_pad",
    )
    assert tuple(candidate["lane_order"]) == (
        "DIRECT_V9",
        "SURFACE_PAD_A",
        "SURFACE_PAD_B",
        "SURFACE_PAD_C",
    )
    assert candidate["lanes"]["DIRECT_V9"] == {
        "dimension": 5,
        "sobol_seed": 20260820,
        "anchor_pad_ordinal": None,
        "anchor_pad_name": None,
        "maximum_prefix_design_sha256": (
            "465fe40f4bf0bb4b1e659f67f702a0baefdde17579eeaf442059bc3c0a87f537"
        ),
    }
    assert tuple(
        candidate["lanes"][name]["anchor_pad_ordinal"]
        for name in ("SURFACE_PAD_A", "SURFACE_PAD_B", "SURFACE_PAD_C")
    ) == (0, 1, 2)
    assert tuple(
        candidate["lanes"][name]["maximum_prefix_design_sha256"]
        for name in ("SURFACE_PAD_A", "SURFACE_PAD_B", "SURFACE_PAD_C")
    ) == (
        "f21412d91685e8c116cc9ef195fd7baf4e9f3725ec5566abe9ede805371c0816",
        "e694f5be3a6853508f3495eaed3fcbc6988ae569363b85d9675b216f1ab2579b",
        "aaa5a1ca60ae9448b5ccca3b606085666bc4006d51ff94a6c1c8d18d72252aac",
    )
    assert candidate["main_total_attempt_budget"] == 256
    assert candidate["maximum_points_per_lane"] == 128
    assert candidate["proposal_failure_consumes_attempt"] is True
    assert candidate["duplicate_consumes_attempt"] is True
    assert candidate["replacement_sampling_allowed"] is False
    assert candidate["local_refinement"]["evaluation_budget"] == 0
    assert candidate["local_refinement"]["execution_status"] == (
        "DISABLED_FOR_V1"
    )
    assert "selection" not in candidate
    ranking = method["post_generation_ranking"]
    assert ranking["implementation_type_id"] == (
        "kcg_connector.grasp.robust.post_generation_ranker."
        "PostGenerationRankOnlyPipeline"
    )
    assert ranking["method_id"] == POST_GENERATION_RANKER_METHOD_ID
    assert ranking["selection"] == "LEXICOGRAPHIC"
    assert ranking["common_scenarios"] == {
        "method_id": POST_GENERATION_SCENARIO_METHOD_ID,
        "source": "SHARED_UNCERTAINTY_PROTOCOL",
        "identical_realizations_for_every_candidate": True,
        "candidate_specific_resampling_allowed": False,
        "scenario_design": "SCRAMBLED_SOBOL",
        "scipy_version": "1.8.0",
        "dimension": POST_GENERATION_SCENARIO_DIMENSION,
        "scenario_count": POST_GENERATION_SCENARIO_COUNT,
        "sobol_seed": POST_GENERATION_SCENARIO_SOBOL_SEED,
        "scramble": True,
        "optimization": None,
        "identity_encoding": "BIG_ENDIAN_BINARY64_ROW_MAJOR",
        "design_sha256": POST_GENERATION_SCENARIO_DESIGN_SHA256,
    }
    assert ranking["failure_retention"] == {
        "retain_every_generation_attempt": True,
        "retain_every_unique_accepted_candidate": True,
        "retain_every_generation_and_evaluation_failure": True,
        "failure_reason_or_exception_required": True,
        "failed_candidate_drop_allowed": False,
        "collision_invocations_per_unique_accepted_candidate": 1,
        "wrench_invocations_per_unique_accepted_candidate": 1,
        "retry_allowed": False,
        "replacement_after_failure_allowed": False,
    }
    assert ranking["formal_selection"] == {
        "status": (
            "EMPTY_UNTIL_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION_"
            "COMPLETE_COLLISION_AND_CALIBRATED_FULL_UNCERTAINTY"
        ),
        "allowed_with_current_bindings": False,
        "formal_ranked_keys_must_be_empty": True,
        "selected_candidate_must_be_none": True,
        "required_collision_claim_scope": COMPLETE_CLEARANCE_SCOPE,
        "current_uncertainty_claim_scope": (
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ),
        "required_additional_uncertainty_binding": (
            "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS"
        ),
    }
    uncertainty = method["uncertainty"]
    assert "independent_validation_scenario_count" not in uncertainty
    assert uncertainty["nested_convergence_scenario_count"] == 256
    base_design = deterministic_sobol(
        dimension=POST_GENERATION_SCENARIO_DIMENSION,
        count=POST_GENERATION_SCENARIO_COUNT,
        seed=POST_GENERATION_SCENARIO_SOBOL_SEED,
    )
    assert hashlib.sha256(
        np.asarray(base_design, dtype=">f8").tobytes(order="C")
    ).hexdigest() == POST_GENERATION_SCENARIO_DESIGN_SHA256
    nested_design = deterministic_sobol(
        dimension=POST_GENERATION_SCENARIO_DIMENSION,
        count=256,
        seed=POST_GENERATION_SCENARIO_SOBOL_SEED,
    )
    assert np.array_equal(
        base_design,
        nested_design[:POST_GENERATION_SCENARIO_COUNT],
    )

    expected_inputs = {
        "config.shared_method",
        "config.hand_contract",
        "config.object_contract",
        "hand.urdf",
        "hand.pad_source_manifest",
        "method.interval_backend_dependency_manifest",
        "method.candidate_qmc_python_dependency_manifest",
        "method.candidate_qmc_ros_dependency_manifest",
        "hand.finger_1_pad.finite_footprint",
        "hand.finger_2_pad.finite_footprint",
        "hand.finger_3_pad.finite_footprint",
        f"objects.{DEVELOPMENT}.planning_geometry",
        f"objects.{DEVELOPMENT}.planning_geometry_manifest",
        f"objects.{DEVELOPMENT}.source_stage",
        f"objects.{DEVELOPMENT}.physical_source",
        f"objects.{DEVELOPMENT}.contact_material_source",
        f"objects.{TRANSFER}.planning_geometry",
        f"objects.{TRANSFER}.original_cad",
        f"objects.{TRANSFER}.geometry_audit",
        f"objects.{TRANSFER}.mass_source",
        f"objects.{TRANSFER}.contact_material_source",
    }
    assert set(audit.input_files) == expected_inputs
    assert set(audit.input_file_sha256) == expected_inputs
    assert audit.input_file_sha256["config.shared_method"] == _sha256(METHOD)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in audit.input_file_sha256.values()
    )
    expected_material_sha256 = (
        "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
    )
    assert audit.input_files[
        f"objects.{DEVELOPMENT}.contact_material_source"
    ].declared_sha256 == expected_material_sha256
    assert audit.input_files[
        f"objects.{TRANSFER}.contact_material_source"
    ].declared_sha256 == expected_material_sha256
    assert audit.derived_geometry_sha256[
        f"objects.{TRANSFER}.original_step_asset"
    ] == "27a8125236894d01d48ac12f8f777a0951f1926c5ed62a210d05f092d575301c"
    assert audit.derived_geometry_sha256[
        f"objects.{DEVELOPMENT}.planning_geometry_asset"
    ] == "ff3dea949aa5c2f320bd4c2907d78fa86a5930cbbbdf3739f50d7f4a1848201e"

    with pytest.raises(TypeError):
        audit.input_file_sha256["new"] = "0" * 64
    with pytest.raises(TypeError):
        audit.canonical_manifest["status"] = "FROZEN"


def test_same_semantic_content_is_yaml_order_independent(tmp_path: Path) -> None:
    reference = _audit()
    reordered_method = _write_yaml(
        tmp_path,
        "method.yaml",
        _reverse_mapping_order(_document(METHOD)),
    )
    reordered_hand = _write_yaml(
        tmp_path,
        "hand.yaml",
        _reverse_mapping_order(_document(HAND)),
    )
    reordered_objects = _write_yaml(
        tmp_path,
        "objects.yaml",
        _reverse_mapping_order(_document(OBJECTS)),
    )
    reordered = _audit(reordered_method, reordered_hand, reordered_objects)

    assert reordered.canonical_json_bytes == reference.canonical_json_bytes
    assert reordered.canonical_sha256 == reference.canonical_sha256
    assert reordered.input_file_sha256["config.shared_method"] != (
        reference.input_file_sha256["config.shared_method"]
    )


def test_one_semantic_field_change_changes_canonical_hash(tmp_path: Path) -> None:
    reference = _audit()
    document = _document(OBJECTS)
    document["objects"][DEVELOPMENT]["dynamic_eligibility"]["reason"] += "_R"
    changed = _audit(objects=_write_yaml(tmp_path, "objects.yaml", document))

    assert changed.canonical_sha256 != reference.canonical_sha256
    assert changed.derived_geometry_sha256 == reference.derived_geometry_sha256


@pytest.mark.parametrize(
    ("source", "duplicate_line"),
    (
        (METHOD, "method: CARTS-Grasp"),
        (HAND, "method: CARTS-Grasp"),
        (OBJECTS, "study_id: CARTS-GRASP-CROSS-OBJECT-V1"),
    ),
)
def test_duplicate_yaml_keys_fail_closed(
    tmp_path: Path, source: Path, duplicate_line: str
) -> None:
    duplicate = tmp_path / source.name
    duplicate.write_text(
        source.read_text(encoding="utf-8") + "\n" + duplicate_line + "\n",
        encoding="utf-8",
    )
    args = {
        METHOD: (duplicate, HAND, OBJECTS),
        HAND: (METHOD, duplicate, OBJECTS),
        OBJECTS: (METHOD, HAND, duplicate),
    }[source]
    with pytest.raises(StudyContractError, match="duplicate YAML key"):
        _audit(*args)


def test_unknown_and_missing_fields_fail_closed(tmp_path: Path) -> None:
    unknown = _document(METHOD)
    unknown["candidate_optimization"]["unregistered_score_weight"] = 0.5
    with pytest.raises(StudyContractError, match="extra=.*unregistered_score_weight"):
        _audit(method=_write_yaml(tmp_path, "unknown.yaml", unknown))

    missing = _document(METHOD)
    del missing["linear_program"]["constraint_scaling"]
    with pytest.raises(StudyContractError, match="missing=.*constraint_scaling"):
        _audit(method=_write_yaml(tmp_path, "missing.yaml", missing))

    missing_lane = _document(METHOD)
    del missing_lane["candidate_optimization"]["lanes"]["SURFACE_PAD_C"][
        "maximum_prefix_design_sha256"
    ]
    with pytest.raises(
        StudyContractError,
        match="missing=.*maximum_prefix_design_sha256",
    ):
        _audit(method=_write_yaml(tmp_path, "missing_lane.yaml", missing_lane))


@pytest.mark.parametrize(
    "legacy_field",
    (
        "continuous_refinement_multistarts",
        "maximum_solver_iterations",
        "relative_objective_tolerance",
        "sobol_seed",
    ),
)
def test_legacy_candidate_optimizer_fields_fail_closed(
    tmp_path: Path,
    legacy_field: str,
) -> None:
    document = _document(METHOD)
    candidate = document["candidate_optimization"]
    assert legacy_field not in candidate
    candidate[legacy_field] = 1
    with pytest.raises(StudyContractError, match=f"extra=.*{legacy_field}"):
        _audit(method=_write_yaml(tmp_path, "legacy_candidate.yaml", document))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            ("candidate_optimization", "method_id"),
            "UNREGISTERED_TOP_GENERATOR",
            "candidate_optimization.method_id",
        ),
        (
            (
                "candidate_optimization",
                "v9_certifier",
                "implementation_type_id",
            ),
            "example.FakeV9",
            "v9_certifier.implementation_type_id",
        ),
        (
            ("candidate_optimization", "v9_certifier", "method_id"),
            "UNREGISTERED_V9",
            "v9_certifier.method_id",
        ),
        (
            (
                "candidate_optimization",
                "v9_certifier",
                "parameter_domain_id",
            ),
            "UNREGISTERED_V9_DOMAIN",
            "v9_certifier.parameter_domain_id",
        ),
        (
            (
                "candidate_optimization",
                "fixed_anchor_mapper",
                "method_id",
            ),
            "UNREGISTERED_FIXED_ANCHOR",
            "fixed_anchor_mapper.method_id",
        ),
        (
            (
                "candidate_optimization",
                "fixed_anchor_mapper",
                "parameter_domain_id",
            ),
            "UNREGISTERED_FIXED_DOMAIN",
            "fixed_anchor_mapper.parameter_domain_id",
        ),
        (
            ("candidate_optimization", "hand_binding", "prepared_pad_order"),
            ["finger_2_pad", "finger_1_pad", "finger_3_pad"],
            "anchor_pad_name|prepared_pad_order",
        ),
        (
            ("candidate_optimization", "lane_order"),
            ["DIRECT_V9", "SURFACE_PAD_B", "SURFACE_PAD_A", "SURFACE_PAD_C"],
            "lane_order",
        ),
        (
            ("candidate_optimization", "lanes", "DIRECT_V9", "dimension"),
            6,
            "DIRECT_V9.dimension",
        ),
        (
            ("candidate_optimization", "lanes", "SURFACE_PAD_A", "sobol_seed"),
            20260822,
            "SURFACE_PAD_A.sobol_seed",
        ),
        (
            (
                "candidate_optimization",
                "lanes",
                "SURFACE_PAD_B",
                "anchor_pad_ordinal",
            ),
            0,
            "SURFACE_PAD_B.anchor_pad_ordinal",
        ),
        (
            (
                "candidate_optimization",
                "lanes",
                "SURFACE_PAD_C",
                "anchor_pad_name",
            ),
            "finger_2_pad",
            "SURFACE_PAD_C.anchor_pad_name",
        ),
        (
            (
                "candidate_optimization",
                "lanes",
                "SURFACE_PAD_C",
                "maximum_prefix_design_sha256",
            ),
            "0" * 64,
            "maximum_prefix_design_sha256",
        ),
        (
            ("candidate_optimization", "schedule_rule"),
            "SEQUENTIAL_LANES",
            "schedule_rule",
        ),
        (
            ("candidate_optimization", "allowed_total_attempt_budgets"),
            [128, 256],
            "allowed_total_attempt_budgets",
        ),
        (
            ("candidate_optimization", "main_total_attempt_budget"),
            512,
            "main_total_attempt_budget",
        ),
        (
            ("candidate_optimization", "maximum_points_per_lane"),
            127,
            "maximum_points_per_lane",
        ),
        (
            ("candidate_optimization", "proposal_failure_consumes_attempt"),
            False,
            "proposal_failure_consumes_attempt",
        ),
        (
            ("candidate_optimization", "duplicate_consumes_attempt"),
            False,
            "duplicate_consumes_attempt",
        ),
        (
            ("candidate_optimization", "replacement_sampling_allowed"),
            True,
            "replacement_sampling_allowed",
        ),
        (
            ("candidate_optimization", "deduplication_rule"),
            "ROUNDED_DECIMAL_VALUES",
            "deduplication_rule",
        ),
        (
            ("candidate_optimization", "sobol_design", "scipy_version"),
            "1.9.0",
            "sobol_design.scipy_version",
        ),
        (
            ("candidate_optimization", "sobol_design", "scramble"),
            False,
            "sobol_design.scramble",
        ),
        (
            ("candidate_optimization", "sobol_design", "optimization"),
            "random-cd",
            "sobol_design.optimization",
        ),
        (
            (
                "candidate_optimization",
                "sobol_design",
                "realized_design_hashes_are_contract_bound",
            ),
            False,
            "realized_design_hashes_are_contract_bound",
        ),
        (
            (
                "candidate_optimization",
                "local_refinement",
                "execution_status",
            ),
            "ENABLED",
            "local_refinement.execution_status",
        ),
        (
            (
                "candidate_optimization",
                "local_refinement",
                "evaluation_budget",
            ),
            1,
            "local_refinement.evaluation_budget",
        ),
        (
            ("candidate_optimization", "external_lane_registry_supported"),
            True,
            "external_lane_registry_supported",
        ),
        (
            (
                "candidate_optimization",
                "legacy_grasp_optimizer_formal_eligible",
            ),
            True,
            "legacy_grasp_optimizer_formal_eligible",
        ),
        (
            ("candidate_optimization", "output_claim"),
            "PHYSICAL_GRASP_ACCEPTED",
            "output_claim",
        ),
    ),
)
def test_top_level_candidate_contract_drift_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    document = _document(METHOD)
    parent: dict[str, Any] = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(StudyContractError, match=message):
        _audit(method=_write_yaml(tmp_path, "candidate_drift.yaml", document))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            ("implementation_type_id",),
            "example.FakePostGenerationRanker",
            "implementation_type_id",
        ),
        (("method_id",), "UNREGISTERED_RANKER", "method_id"),
        (("selection",), "WEIGHTED_SUM", "selection"),
        (
            ("selection_order",),
            [
                "qmc_lower_tail_mean_task_margin",
                "hard_bound_minimum_task_margin",
                "minimum_peak_normal_force_n",
                "minimum_joint_torque_utilization",
                "maximum_trajectory_clearance_m",
            ],
            "selection_order",
        ),
        (("common_scenarios", "source"), "LOCAL", "common_scenarios.source"),
        (
            ("common_scenarios", "identical_realizations_for_every_candidate"),
            False,
            "identical_realizations_for_every_candidate",
        ),
        (
            ("common_scenarios", "candidate_specific_resampling_allowed"),
            True,
            "candidate_specific_resampling_allowed",
        ),
        (
            ("common_scenarios", "method_id"),
            "UNREGISTERED_COMMON_DESIGN",
            "common_scenarios.method_id",
        ),
        (("common_scenarios", "dimension"), 2, "dimension"),
        (("common_scenarios", "scenario_count"), 129, "scenario_count"),
        (("common_scenarios", "sobol_seed"), 20260821, "sobol_seed"),
        (
            ("common_scenarios", "design_sha256"),
            "0" * 64,
            "design_sha256",
        ),
        (
            (
                "failure_retention",
                "retain_every_generation_and_evaluation_failure",
            ),
            False,
            "retain_every_generation_and_evaluation_failure",
        ),
        (
            (
                "failure_retention",
                "failure_reason_or_exception_required",
            ),
            False,
            "failure_reason_or_exception_required",
        ),
        (
            ("failure_retention", "replacement_after_failure_allowed"),
            True,
            "replacement_after_failure_allowed",
        ),
        (
            ("formal_selection", "required_collision_claim_scope"),
            "CLOSURE_ONLY",
            "required_collision_claim_scope",
        ),
        (
            ("formal_selection", "current_uncertainty_claim_scope"),
            "ALL_UNCERTAINTIES_CERTIFIED",
            "current_uncertainty_claim_scope",
        ),
    ),
)
def test_post_generation_ranking_protocol_drift_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    document = _document(METHOD)
    parent = document["post_generation_ranking"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(StudyContractError, match=message):
        _audit(method=_write_yaml(tmp_path, "ranking_drift.yaml", document))


def test_post_generation_binding_missing_or_extra_fields_fail_closed(
    tmp_path: Path,
) -> None:
    missing = _document(METHOD)
    del missing["post_generation_ranking"]["implementation_type_id"]
    with pytest.raises(
        StudyContractError,
        match="missing=.*implementation_type_id",
    ):
        _audit(method=_write_yaml(tmp_path, "ranking_missing.yaml", missing))

    extra = _document(METHOD)
    extra["post_generation_ranking"]["common_scenarios"][
        "unregistered_design_alias"
    ] = "0" * 64
    with pytest.raises(
        StudyContractError,
        match="extra=.*unregistered_design_alias",
    ):
        _audit(method=_write_yaml(tmp_path, "ranking_extra.yaml", extra))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("nested_convergence_scenario_count", 128),
        ("nested_convergence_role", "INDEPENDENT_VALIDATION"),
    ),
)
def test_nested_convergence_cannot_be_relabelled_as_independent_validation(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    document = _document(METHOD)
    document["uncertainty"][field] = value
    with pytest.raises(StudyContractError, match=field):
        _audit(method=_write_yaml(tmp_path, "nested_drift.yaml", document))

    legacy = _document(METHOD)
    legacy["uncertainty"]["independent_validation_scenario_count"] = 256
    with pytest.raises(
        StudyContractError,
        match="extra=.*independent_validation_scenario_count",
    ):
        _audit(method=_write_yaml(tmp_path, "false_independent.yaml", legacy))


def test_object_specific_algorithm_override_fails_before_generic_schema_error(
    tmp_path: Path,
) -> None:
    document = _document(OBJECTS)
    document["objects"][TRANSFER]["algorithm_override"] = {
        "candidate_budget": 4096,
    }
    with pytest.raises(
        StudyContractError, match="object-specific algorithm override"
    ):
        _audit(objects=_write_yaml(tmp_path, "override.yaml", document))


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda value: value["transfer_protocol"].__setitem__(
                "transfer_object_geometry_seen_before_method_freeze", False
            ),
            "transfer_object_geometry_seen_before_method_freeze",
        ),
        (
            lambda value: value["transfer_protocol"].__setitem__(
                "prospective_double_blind_claim_allowed", True
            ),
            "prospective_double_blind_claim_allowed",
        ),
        (
            lambda value: value["transfer_protocol"].__setitem__(
                "frozen_transfer_object", DEVELOPMENT
            ),
            "frozen_transfer_object",
        ),
        (
            lambda value: value["objects"][TRANSFER].__setitem__(
                "role", "PROSPECTIVE_DOUBLE_BLIND_HOLDOUT"
            ),
            "extra=.*role",
        ),
    ),
)
def test_holdout_role_deception_fails_closed(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    document = _document(OBJECTS)
    mutator(document)
    with pytest.raises(StudyContractError, match=message):
        _audit(objects=_write_yaml(tmp_path, "deception.yaml", document))


def test_dynamic_eligibility_cannot_be_upgraded(tmp_path: Path) -> None:
    document = _document(OBJECTS)
    document["objects"][TRANSFER]["dynamic_eligibility"] = {
        "allowed": True,
        "reason": "READY",
    }
    with pytest.raises(StudyContractError, match="dynamic_eligibility.allowed"):
        _audit(objects=_write_yaml(tmp_path, "dynamic.yaml", document))


def test_transfer_material_interval_cannot_drift_from_development(
    tmp_path: Path,
) -> None:
    document = _document(OBJECTS)
    document["objects"][TRANSFER]["contact_material_uncertainty"][
        "friction_coefficient"
    ] = [0.45, 0.56]
    with pytest.raises(StudyContractError, match="one shared friction interval"):
        _audit(objects=_write_yaml(tmp_path, "material_drift.yaml", document))


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (
            "post_generation_ranking",
            "selection",
            "WEIGHTED_SUM",
            "selection",
        ),
        (
            "linear_program",
            "constraint_scaling",
            "HIGHS_AUTOMATIC_ONLY",
            "constraint_scaling",
        ),
        (
            "contact_model",
            "friction_cone_approximation",
            "OUTER_POLYGON",
            "friction_cone_approximation",
        ),
    ),
)
def test_method_ranking_scaling_and_friction_cone_cannot_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
    message: str,
) -> None:
    document = _document(METHOD)
    document[section][field] = value
    with pytest.raises(StudyContractError, match=message):
        _audit(method=_write_yaml(tmp_path, "method.yaml", document))


def test_online_truth_firewall_and_closure_actuation_cannot_drift(
    tmp_path: Path,
) -> None:
    method = _document(METHOD)
    method["shared_protocol"]["online_contact_truth_allowed"] = True
    with pytest.raises(StudyContractError, match="online_contact_truth_allowed"):
        _audit(method=_write_yaml(tmp_path, "truth.yaml", method))

    hand = copy.deepcopy(_document(HAND))
    hand["closure_actuation"]["rows"]["f1"]["joint_weights"] = {"f1j1": 1.0}
    with pytest.raises(StudyContractError, match="shared or foreign"):
        _audit(hand=_write_yaml(tmp_path, "hand.yaml", hand))


def test_rayclosure_missing_and_extra_fields_fail_closed(tmp_path: Path) -> None:
    missing = _document(METHOD)
    del missing["ray_closure"]["closure_parameter_domain"]["dimension"]
    with pytest.raises(StudyContractError, match="missing=.*dimension"):
        _audit(method=_write_yaml(tmp_path, "missing_ray.yaml", missing))

    extra = _document(METHOD)
    extra["ray_closure"]["physical_margin_threshold"] = 0.01
    with pytest.raises(StudyContractError, match="extra=.*physical_margin_threshold"):
        _audit(method=_write_yaml(tmp_path, "extra_ray.yaml", extra))

    missing_interval = _document(METHOD)
    del missing_interval["ray_closure"]["interval_backend"]["decimal_precision"]
    with pytest.raises(StudyContractError, match="missing=.*decimal_precision"):
        _audit(
            method=_write_yaml(
                tmp_path,
                "missing_interval.yaml",
                missing_interval,
            )
        )

    extra_interval = _document(METHOD)
    extra_interval["ray_closure"]["interval_backend"][
        "physical_contact_tolerance_m"
    ] = 1.0e-6
    with pytest.raises(
        StudyContractError,
        match="extra=.*physical_contact_tolerance_m",
    ):
        _audit(
            method=_write_yaml(
                tmp_path,
                "extra_interval.yaml",
                extra_interval,
            )
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            ("ray_closure", "method_id"),
            "UNREGISTERED_RAY_CLOSURE",
            "method_id",
        ),
        (
            ("ray_closure", "maximum_subdivision_intervals"),
            2048,
            "maximum_subdivision_intervals",
        ),
        (
            ("ray_closure", "maximum_subdivision_intervals_role"),
            "PHYSICAL_ACCEPTANCE_THRESHOLD",
            "maximum_subdivision_intervals_role",
        ),
        (
            ("ray_closure", "subdivision_budget_exhaustion_policy"),
            "ACCEPT_PARTIAL_CERTIFICATE",
            "subdivision_budget_exhaustion_policy",
        ),
        (
            ("ray_closure", "budget_convergence_values"),
            [2048, 4096, 8192],
            "budget_convergence_values",
        ),
        (
            ("ray_closure", "physical_acceptance_gate"),
            True,
            "physical_acceptance_gate",
        ),
        (
            ("ray_closure", "interval_backend", "method_id"),
            "UNREGISTERED_INTERVAL_BACKEND",
            "interval_backend.method_id",
        ),
        (
            ("ray_closure", "interval_backend", "decimal_precision"),
            32,
            "interval_backend.decimal_precision",
        ),
        (
            (
                "ray_closure",
                "interval_backend",
                "maximum_root_bisection_iterations",
            ),
            1024,
            "maximum_root_bisection_iterations",
        ),
        (
            (
                "ray_closure",
                "interval_backend",
                "root_budget_exhaustion_policy",
            ),
            "ACCEPT_LAST_INTERVAL",
            "root_budget_exhaustion_policy",
        ),
        (
            ("ray_closure", "interval_backend", "physical_acceptance_gate"),
            True,
            "interval_backend.physical_acceptance_gate",
        ),
        (
            ("ray_closure", "closure_parameter_domain", "domain_id"),
            "UNREGISTERED_DOMAIN",
            "domain_id",
        ),
        (
            ("ray_closure", "closure_parameter_domain", "dimension"),
            4,
            "parameter_layout.*match dimension",
        ),
        (
            (
                "ray_closure",
                "closure_parameter_domain",
                "assembly_axis_yaw",
                "unit_interval",
            ),
            "CLOSED_ZERO_TO_ONE",
            "assembly_axis_yaw.unit_interval",
        ),
        (
            (
                "ray_closure",
                "closure_parameter_domain",
                "placement",
                "unit_interval",
            ),
            "HALF_OPEN_ZERO_TO_ONE",
            "placement.unit_interval",
        ),
        (
            (
                "ray_closure",
                "closure_parameter_domain",
                "preshape",
                "physical_bounds_source",
            ),
            "OBJECT_SPECIFIC_GUESSED_LIMITS",
            "preshape.physical_bounds_source",
        ),
    ),
)
def test_rayclosure_illegal_values_fail_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    document = _document(METHOD)
    parent: dict[str, Any] = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(StudyContractError, match=message):
        _audit(method=_write_yaml(tmp_path, "illegal_ray.yaml", document))


def test_rayclosure_layout_must_match_hash_bound_hand_derivation(
) -> None:
    document = _document(METHOD)
    domain = document["ray_closure"]["closure_parameter_domain"]
    domain["parameter_layout"][-1] = "preshape_joint_unit:f2j1"
    domain["preshape"]["joint_names"] = ["f2j1"]
    hand_contract = load_carts_hand_contract(HAND, repository_root=REPOSITORY)
    with pytest.raises(StudyContractError, match="derived from hand"):
        _validate_ray_closure_hand_binding(document, hand_contract)


@pytest.mark.parametrize("binding", ("preshape", "pad_order"))
def test_top_generator_layout_and_pad_order_are_derived_from_hand(
    binding: str,
) -> None:
    document = _document(METHOD)
    candidate = document["candidate_optimization"]
    if binding == "preshape":
        candidate["hand_binding"]["preshape_joint_names"] = ["f2j1"]
        candidate["v9_certifier"]["parameter_layout"][-1] = (
            "preshape_joint_unit:f2j1"
        )
        candidate["fixed_anchor_mapper"]["parameter_layout"][-1] = (
            "preshape_joint_unit:f2j1"
        )
    else:
        candidate["hand_binding"]["prepared_pad_order"] = [
            "finger_2_pad",
            "finger_1_pad",
            "finger_3_pad",
        ]
        candidate["lanes"]["SURFACE_PAD_A"]["anchor_pad_name"] = (
            "finger_2_pad"
        )
        candidate["lanes"]["SURFACE_PAD_B"]["anchor_pad_name"] = (
            "finger_1_pad"
        )
    hand_contract = load_carts_hand_contract(HAND, repository_root=REPOSITORY)
    with pytest.raises(StudyContractError, match="derived from hand"):
        _validate_ray_closure_hand_binding(document, hand_contract)


def test_interval_backend_dependency_manifest_hash_is_verified(
    tmp_path: Path,
) -> None:
    document = _document(METHOD)
    document["ray_closure"]["interval_backend"]["dependency"][
        "manifest_sha256"
    ] = "0" * 64
    with pytest.raises(StudyContractError, match="manifest_sha256"):
        _audit(method=_write_yaml(tmp_path, "bad_interval_sha.yaml", document))


@pytest.mark.parametrize(
    "field",
    ("python_manifest_sha256", "ros_manifest_sha256"),
)
def test_candidate_qmc_dependency_manifest_hashes_are_verified(
    tmp_path: Path,
    field: str,
) -> None:
    document = _document(METHOD)
    document["candidate_optimization"]["sobol_design"]["dependency"][field] = (
        "0" * 64
    )
    with pytest.raises(StudyContractError, match=field):
        _audit(method=_write_yaml(tmp_path, "bad_qmc_sha.yaml", document))


def test_scipy_dependency_is_explicit_in_python_and_ros_manifests() -> None:
    setup_text = (REPOSITORY / "src/kcg_connector/setup.py").read_text(
        encoding="utf-8"
    )
    package_text = (REPOSITORY / "src/kcg_connector/package.xml").read_text(
        encoding="utf-8"
    )
    assert setup_text.count('"scipy==1.8.0"') == 1
    assert package_text.count("<exec_depend>python3-scipy</exec_depend>") == 1


def test_formal_freeze_reports_remaining_non_rayclosure_blockers() -> None:
    with pytest.raises(StudyFreezeIncompleteError) as captured:
        build_frozen_study_contract(
            METHOD,
            HAND,
            OBJECTS,
            repository_root=REPOSITORY,
        )
    assert captured.value.blockers == (
        "MISSING_FORMAL_ROOT_INTERVAL_CANDIDATE_PROPAGATION",
        "MISSING_COMPLETE_HAND_ENVIRONMENT_CONTINUOUS_COLLISION_BINDING",
        "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS",
    )
