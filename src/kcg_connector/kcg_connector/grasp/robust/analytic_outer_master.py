"""Global outer master for the TE continuous-grasp analytic envelope.

This module encodes one deliberately limited mathematical object: a finite-
scenario outer relaxation of the actuator-envelope static wrench margin.  Its
SCIP dual bound is a lower bound for ``J_AE = -gamma_AE``.  A primal point of
this model is not a robust grasp, a controller-reachable state, or an Isaac
success claim.

The compact root spans the declared continuous grasp domain rather than an old
candidate list.  Each object contact point is relaxed to the exact-one union of
the axis-aligned boxes of all non-hard-forbidden STEP parent-face triangle
groups, while every triangle of each complete fingertip pad remains an
exact-one mode with continuous barycentric coordinates.  No parent-face box is
treated as a physical forbidden region.  Each object-side local friction cone
is relaxed to its common force-norm cap intersected with six global
axis-support halfspaces computed over the complete atlas.  A conservative
pad-side unilateral friction envelope additionally requires the reaction to
press into the selected pad;
triangle vertex stars retain edge/vertex normal cones, and overly broad stars
are left unconstrained.  The shared contact geometry is linked through the
URDF kinematics and a full quaternion ``T_HC``.  Only contact force varies
between the frozen finite wrench scenarios.  This root is deliberately loose;
its primal points are not surface contacts or grasp candidates.

The optional axial-endpoint model has a narrower role.  It holds the
hand--object transform fixed during closure and maps the frozen UV charts to
their analytic plane/cylinder carriers.  The controller contract permits
passive hand--object motion while contact is established, and those analytic
carriers are not a proved superset of the frozen mesh.  That model is therefore
only a zero-relative-motion mode-seed screen; neither its timeout nor a future
infeasibility result may prune the controller-reachable grasp domain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.hand_contract import (
    CARTSHandContract,
    VerifiedPad,
    load_carts_hand_contract,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.surface_atlas import (
    StepContactAtlas,
    load_step_contact_atlas,
)


SCHEMA_VERSION = "kcg_te_continuous_grasp_analytic_envelope_v1"
CLAIM_SCOPE = "SIMULATION_ONLY_ACTUATOR_ENVELOPE_NOT_PD_OR_DYNAMIC_CAPACITY"
EXPECTED_DEVELOPMENT_OBJECT = "te_deutsch_d38999_26fj35pn_step"
EXPECTED_PAD_TRIANGLES = 2442
EXPECTED_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
EXPECTED_CLOSING_JOINTS = ("f1j2", "f2j1", "f3j2")
EXPECTED_CLOSING_ORDERS = (
    ("finger_1", "finger_2", "finger_3"),
    ("finger_1", "finger_3", "finger_2"),
    ("finger_2", "finger_1", "finger_3"),
    ("finger_2", "finger_3", "finger_1"),
    ("finger_3", "finger_1", "finger_2"),
    ("finger_3", "finger_2", "finger_1"),
)
EXPECTED_OMISSIONS = (
    "continuous_pose_uncertainty",
    "exact_pd_equation_after_nominal_gravity",
    "required_closing_effort_preload_closed_loop_response",
    "exact_object_surface_membership_and_contact_normal_coupling",
    "pad_object_normal_opposition",
    "contact_nonpenetration_and_complementarity",
    "forbidden_surface_clearance",
    "hand_self_object_and_table_collision",
    "closure_order_contact_timing_and_pad_before_nonpad",
    "arm_ik_and_continuous_paths",
)


class AnalyticOuterMasterError(ValueError):
    """Raised when the frozen input or mathematical outer model is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise AnalyticOuterMasterError("YAML keys must be hashable") from error
        if duplicate:
            raise AnalyticOuterMasterError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AnalyticOuterMasterError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected).difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise AnalyticOuterMasterError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )


def _exact_string(value: Any, expected: str, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise AnalyticOuterMasterError(f"{label} must be exactly {expected!r}")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise AnalyticOuterMasterError(f"{label} must be exactly {expected}")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnalyticOuterMasterError(f"{label} must be positive and finite")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise AnalyticOuterMasterError(f"{label} must be positive and finite")
    return parsed


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AnalyticOuterMasterError(f"{label} must contain {length} numbers")
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise AnalyticOuterMasterError(f"{label} must contain {length} finite numbers")
    return result


def _sequence_of_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AnalyticOuterMasterError(f"{label} must be a sequence of strings")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise AnalyticOuterMasterError(f"{label} must be a sequence of strings")
    return result


def _repository_file(root: Path, value: Any, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise AnalyticOuterMasterError(f"{label} must be repository-relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AnalyticOuterMasterError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnalyticEnvelopeContract:
    path: Path
    repository_root: Path
    step_contact_atlas_path: Path
    hand_contact_contract_path: Path
    mass_kg: float
    center_of_mass_object_m: tuple[float, float, float]
    inertia_object_kg_m2: tuple[tuple[float, float, float], ...]
    gravity_object_m_s2: tuple[float, float, float]
    visual_pose_empirical_hard_set_defined: bool
    visual_pose_empirical_hard_set_status: str
    visual_pose_empirical_hard_set_source: str
    visual_pose_measurement_protocol_executable: bool
    visual_pose_measurement_protocol_status: str
    visual_pose_observation_rule: str
    visual_pose_estimate_truth_firewall: str
    visual_pose_residual_coordinates: tuple[str, ...]
    visual_pose_residual_frame: str
    visual_pose_residual_transform: str
    visual_pose_orientation_branch_rule: str
    visual_pose_failure_rule: str
    visual_pose_empirical_support_rule: str
    visual_pose_miss_rule: str
    visual_pose_empirical_support_claim: str
    visual_pose_truth_use: str
    visual_pose_research_set_defined: bool
    visual_pose_research_set_frozen_before_robust_grasp_outcomes: bool
    visual_pose_research_set_role: str
    visual_pose_research_set_source: str
    visual_pose_research_set_coordinates: str
    visual_pose_research_set_lower: tuple[float, ...]
    visual_pose_research_set_upper: tuple[float, ...]
    visual_pose_research_set_guarantee_scope: str
    visual_pose_research_set_transfer_limit: str
    stress_test_pose_translation_radius_m: float
    stress_test_pose_rotation_radius_rad: float
    stress_test_pose_role: str
    friction_interval: tuple[float, float]
    wrench_force_radius_n: float
    wrench_torque_radius_nm: float
    characteristic_radius_m: float
    contact_count: int
    normal_force_cap_n: float
    independent_joint_names: tuple[str, ...]
    closing_joint_names: tuple[str, ...]
    closing_orders: tuple[tuple[str, str, str], ...]
    active_finger_first_contact_rule: str
    active_finger_pad_triangle_count: int
    active_finger_nonpad_triangle_count: int
    hand_drive_maximum_effort_nm: float
    contact_effort_rise_nm: float
    measured_effort_abort_nm: float
    required_closing_joint_effort_role: str
    hand_stiffness_nm_per_rad: float
    hand_damping_nm_s_per_rad: float
    friction_outer_polygon_edges: int
    omitted_for_outer_relaxation: tuple[str, ...]

    @property
    def friction_minimum(self) -> float:
        return self.friction_interval[0]

    @property
    def outer_single_contact_force_norm_cap_n(self) -> float:
        mu_polygon = self.friction_minimum / math.cos(
            math.pi / self.friction_outer_polygon_edges
        )
        return self.normal_force_cap_n * math.sqrt(1.0 + mu_polygon * mu_polygon)

    @property
    def force_only_gamma_upper_bound(self) -> float:
        gravity_force = self.mass_kg * float(
            np.linalg.norm(self.gravity_object_m_s2)
        )
        # The frozen +gravity-direction disturbance requires
        # (1 + gamma) * m*g of resultant contact force.
        return (
            self.contact_count * self.outer_single_contact_force_norm_cap_n
            / gravity_force
            - 1.0
        )


def load_analytic_envelope_contract(
    contract_path: str | Path, *, repository_root: str | Path
) -> AnalyticEnvelopeContract:
    root = Path(repository_root).resolve(strict=True)
    supplied = Path(contract_path)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"analytic envelope contract is unavailable: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise AnalyticOuterMasterError("cannot load analytic envelope YAML") from error
    root_value = _mapping(document, "analytic envelope root")
    _exact_keys(
        root_value,
        (
            "schema_version",
            "claim_scope",
            "hardware_authorized",
            "development_object",
            "inputs",
            "object_model",
            "continuous_uncertainty",
            "contact_and_actuation",
            "outer_master",
        ),
        "analytic envelope root",
    )
    _exact_string(root_value["schema_version"], SCHEMA_VERSION, "schema_version")
    _exact_string(root_value["claim_scope"], CLAIM_SCOPE, "claim_scope")
    _exact_bool(root_value["hardware_authorized"], False, "hardware_authorized")
    _exact_string(
        root_value["development_object"],
        EXPECTED_DEVELOPMENT_OBJECT,
        "development_object",
    )

    inputs = _mapping(root_value["inputs"], "inputs")
    _exact_keys(inputs, ("step_contact_atlas", "hand_contact_contract"), "inputs")
    atlas_path = _repository_file(root, inputs["step_contact_atlas"], "step atlas")
    hand_path = _repository_file(
        root, inputs["hand_contact_contract"], "hand contract"
    )

    object_model = _mapping(root_value["object_model"], "object_model")
    _exact_keys(
        object_model,
        (
            "mass_kg",
            "center_of_mass_object_m",
            "inertia_object_kg_m2",
            "gravity_object_m_s2",
            "source",
        ),
        "object_model",
    )
    mass = _positive(object_model["mass_kg"], "object_model.mass_kg")
    com = _finite_vector(
        object_model["center_of_mass_object_m"], 3, "object_model.center_of_mass"
    )
    inertia_rows_raw = object_model["inertia_object_kg_m2"]
    if not isinstance(inertia_rows_raw, Sequence) or len(inertia_rows_raw) != 3:
        raise AnalyticOuterMasterError("object inertia must be a 3x3 matrix")
    inertia = tuple(
        _finite_vector(row, 3, f"object inertia row {index}")
        for index, row in enumerate(inertia_rows_raw)
    )
    inertia_array = np.asarray(inertia, dtype=np.float64)
    if not np.allclose(inertia_array, inertia_array.T, rtol=0.0, atol=1.0e-15):
        raise AnalyticOuterMasterError("object inertia must be symmetric")
    if float(np.min(np.linalg.eigvalsh(inertia_array))) <= 0.0:
        raise AnalyticOuterMasterError("object inertia must be positive definite")
    gravity = _finite_vector(
        object_model["gravity_object_m_s2"], 3, "object_model.gravity"
    )
    if float(np.linalg.norm(gravity)) <= 0.0:
        raise AnalyticOuterMasterError("object gravity must be non-zero")
    if not isinstance(object_model["source"], str) or not object_model["source"]:
        raise AnalyticOuterMasterError("object_model.source must be named")

    uncertainty = _mapping(
        root_value["continuous_uncertainty"], "continuous_uncertainty"
    )
    _exact_keys(
        uncertainty,
        (
            "visual_pose_empirical_hard_set_defined",
            "visual_pose_empirical_hard_set_status",
            "visual_pose_empirical_hard_set_source",
            "visual_pose_measurement_protocol_executable",
            "visual_pose_measurement_protocol_status",
            "visual_pose_observation_rule",
            "visual_pose_estimate_truth_firewall",
            "visual_pose_residual_coordinates",
            "visual_pose_residual_frame",
            "visual_pose_residual_transform",
            "visual_pose_orientation_branch_rule",
            "visual_pose_failure_rule",
            "visual_pose_empirical_support_rule",
            "visual_pose_miss_rule",
            "visual_pose_empirical_support_claim",
            "visual_pose_truth_use",
            "visual_pose_research_set_defined",
            "visual_pose_research_set_frozen_before_robust_grasp_outcomes",
            "visual_pose_research_set_role",
            "visual_pose_research_set_source",
            "visual_pose_research_set_coordinates",
            "visual_pose_research_set_lower",
            "visual_pose_research_set_upper",
            "visual_pose_research_set_coverage_probability",
            "visual_pose_research_set_guarantee_scope",
            "visual_pose_research_set_transfer_limit",
            "stress_test_pose_translation_radius_m",
            "stress_test_pose_rotation_radius_rad",
            "stress_test_pose_role",
            "shared_friction_interval",
            "wrench_force_radius_n",
            "wrench_torque_radius_nm",
            "characteristic_radius_m",
            "probability_distribution_claimed",
            "camera_performance_certification_claimed",
        ),
        "continuous_uncertainty",
    )
    visual_pose_empirical_hard_set_defined = _exact_bool(
        uncertainty["visual_pose_empirical_hard_set_defined"],
        False,
        "visual_pose_empirical_hard_set_defined",
    )
    visual_pose_empirical_hard_set_status = _exact_string(
        uncertainty["visual_pose_empirical_hard_set_status"],
        "PENDING_FROZEN_GLOBAL_WRIST_PALM_TE_SIM_6DOF_RESIDUAL_PROTOCOL",
        "visual_pose_empirical_hard_set_status",
    )
    visual_pose_empirical_hard_set_source = _exact_string(
        uncertainty["visual_pose_empirical_hard_set_source"],
        "PENDING_MATCHED_ESTIMATE_AND_POST_HOC_SIM_TRUTH_PAIRS",
        "visual_pose_empirical_hard_set_source",
    )
    visual_pose_measurement_protocol_executable = _exact_bool(
        uncertainty["visual_pose_measurement_protocol_executable"],
        False,
        "visual_pose_measurement_protocol_executable",
    )
    visual_pose_measurement_protocol_status = _exact_string(
        uncertainty["visual_pose_measurement_protocol_status"],
        "SKELETON_FROZEN_EXECUTION_INPUTS_PENDING",
        "visual_pose_measurement_protocol_status",
    )
    visual_pose_observation_rule = _exact_string(
        uncertainty["visual_pose_observation_rule"],
        "SYNCHRONIZED_GLOBAL_WRIST_PALM_RGB_AND_METRIC_DEPTH_PRECONTACT",
        "visual_pose_observation_rule",
    )
    visual_pose_estimate_truth_firewall = _exact_string(
        uncertainty["visual_pose_estimate_truth_firewall"],
        "ESTIMATE_JSON_COMMITTED_BEFORE_POST_HOC_TRUTH_READ",
        "visual_pose_estimate_truth_firewall",
    )
    visual_pose_residual_coordinates = _sequence_of_strings(
        uncertainty["visual_pose_residual_coordinates"],
        "visual pose residual coordinates",
    )
    if visual_pose_residual_coordinates != (
        "dx_m",
        "dy_m",
        "dz_m",
        "rx_rad",
        "ry_rad",
        "rz_rad",
    ):
        raise AnalyticOuterMasterError("visual pose residual coordinates changed")
    visual_pose_residual_frame = _exact_string(
        uncertainty["visual_pose_residual_frame"],
        "ESTIMATED_TE_CONNECTOR_TASK_FRAME",
        "visual_pose_residual_frame",
    )
    visual_pose_residual_transform = _exact_string(
        uncertainty["visual_pose_residual_transform"],
        "RIGHT_INVARIANT_RELATIVE_TRANSLATION_PLUS_SO3_LOG",
        "visual_pose_residual_transform",
    )
    visual_pose_orientation_branch_rule = _exact_string(
        uncertainty["visual_pose_orientation_branch_rule"],
        "NO_POST_HOC_TRUTH_SELECTION_OR_SYMMETRY_MINIMIZATION",
        "visual_pose_orientation_branch_rule",
    )
    visual_pose_failure_rule = _exact_string(
        uncertainty["visual_pose_failure_rule"],
        "NO_OUTPUT_NONFINITE_OR_UNRESOLVED_MULTIMODAL_IS_MISS",
        "visual_pose_failure_rule",
    )
    visual_pose_empirical_support_rule = _exact_string(
        uncertainty["visual_pose_empirical_support_rule"],
        "COMPONENTWISE_ASYMMETRIC_MIN_MAX_OF_ALL_ACCEPTED_FINITE_RESIDUALS_NO_OUTLIER_REMOVAL",
        "visual_pose_empirical_support_rule",
    )
    visual_pose_miss_rule = _exact_string(
        uncertainty["visual_pose_miss_rule"],
        "MISS_RATE_REPORTED_CONTROLLER_ABSTAINS_ANY_PREREGISTERED_SCENE_ALL_MISS_LEAVES_U_UNDEFINED",
        "visual_pose_miss_rule",
    )
    visual_pose_empirical_support_claim = _exact_string(
        uncertainty["visual_pose_empirical_support_claim"],
        "ACCEPTED_CONDITIONAL_SIMULATION_EMPIRICAL_SUPPORT_ONLY",
        "visual_pose_empirical_support_claim",
    )
    visual_pose_truth_use = _exact_string(
        uncertainty["visual_pose_truth_use"],
        "POST_HOC_RESIDUAL_EVALUATION_ONLY",
        "visual_pose_truth_use",
    )
    visual_pose_research_set_defined = _exact_bool(
        uncertainty["visual_pose_research_set_defined"],
        True,
        "visual_pose_research_set_defined",
    )
    visual_pose_research_set_frozen_before_robust_grasp_outcomes = _exact_bool(
        uncertainty["visual_pose_research_set_frozen_before_robust_grasp_outcomes"],
        True,
        "visual_pose_research_set_frozen_before_robust_grasp_outcomes",
    )
    visual_pose_research_set_role = _exact_string(
        uncertainty["visual_pose_research_set_role"],
        "LITERATURE_ANCHORED_EX_ANTE_STUDY_DOMAIN",
        "visual_pose_research_set_role",
    )
    visual_pose_research_set_source = _exact_string(
        uncertainty["visual_pose_research_set_source"],
        "LEE_ET_AL_2026_DOI_10_1007_S12541_026_01488_7_WITH_CONNECTOR_SCALE_SANITY_CHECKS",
        "visual_pose_research_set_source",
    )
    visual_pose_research_set_coordinates = _exact_string(
        uncertainty["visual_pose_research_set_coordinates"],
        "RIGHT_INVARIANT_TRANSLATION_PLUS_SO3_LOG",
        "visual_pose_research_set_coordinates",
    )
    visual_pose_research_set_lower = _finite_vector(
        uncertainty["visual_pose_research_set_lower"],
        6,
        "visual_pose_research_set_lower",
    )
    visual_pose_research_set_upper = _finite_vector(
        uncertainty["visual_pose_research_set_upper"],
        6,
        "visual_pose_research_set_upper",
    )
    expected_research_half_widths = (
        0.002,
        0.002,
        0.002,
        math.radians(2.0),
        math.radians(2.0),
        math.radians(2.0),
    )
    for index, half_width in enumerate(expected_research_half_widths):
        if not math.isclose(
            visual_pose_research_set_lower[index],
            -half_width,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ) or not math.isclose(
            visual_pose_research_set_upper[index],
            half_width,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise AnalyticOuterMasterError(
                "visual pose research set must remain the frozen +/-2 mm, +/-2 degree box"
            )
    if uncertainty["visual_pose_research_set_coverage_probability"] is not None:
        raise AnalyticOuterMasterError(
            "the literature-anchored research set must not claim coverage probability"
        )
    visual_pose_research_set_guarantee_scope = _exact_string(
        uncertainty["visual_pose_research_set_guarantee_scope"],
        "CONDITIONAL_ON_ESTIMATOR_ACCEPTANCE_AND_TRUE_RESIDUAL_INSIDE_SET",
        "visual_pose_research_set_guarantee_scope",
    )
    visual_pose_research_set_transfer_limit = _exact_string(
        uncertainty["visual_pose_research_set_transfer_limit"],
        "LITERATURE_EULER_AND_BASE_FRAME_VALUES_ANCHOR_MAGNITUDE_ONLY_NOT_COORDINATE_EQUIVALENCE_OR_COVERAGE",
        "visual_pose_research_set_transfer_limit",
    )
    stress_translation = _positive(
        uncertainty["stress_test_pose_translation_radius_m"],
        "stress-test pose translation radius",
    )
    stress_rotation = _positive(
        uncertainty["stress_test_pose_rotation_radius_rad"],
        "stress-test pose rotation radius",
    )
    stress_role = _exact_string(
        uncertainty["stress_test_pose_role"],
        "POST_CALIBRATION_STRESS_TEST_ONLY_NOT_VISUAL_ERROR_HARD_SET",
        "stress_test_pose_role",
    )
    friction = _finite_vector(
        uncertainty["shared_friction_interval"], 2, "friction interval"
    )
    if friction[0] < 0.0 or friction[0] > friction[1]:
        raise AnalyticOuterMasterError("friction interval must be nonnegative and ordered")
    force_radius = _positive(
        uncertainty["wrench_force_radius_n"], "wrench force radius"
    )
    torque_radius = _positive(
        uncertainty["wrench_torque_radius_nm"], "wrench torque radius"
    )
    characteristic_radius = _positive(
        uncertainty["characteristic_radius_m"], "characteristic radius"
    )
    _exact_bool(
        uncertainty["probability_distribution_claimed"],
        False,
        "probability_distribution_claimed",
    )
    _exact_bool(
        uncertainty["camera_performance_certification_claimed"],
        False,
        "camera_performance_certification_claimed",
    )
    gravity_force = mass * float(np.linalg.norm(gravity))
    derived_radius = math.sqrt(float(np.trace(inertia_array)) / (2.0 * mass))
    derived_torque = gravity_force * derived_radius
    if not math.isclose(force_radius, gravity_force, rel_tol=0.0, abs_tol=1.0e-12):
        raise AnalyticOuterMasterError("wrench force radius differs from m*g")
    if not math.isclose(
        characteristic_radius, derived_radius, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise AnalyticOuterMasterError("characteristic radius differs from inertia")
    if not math.isclose(torque_radius, derived_torque, rel_tol=0.0, abs_tol=1.0e-12):
        raise AnalyticOuterMasterError("wrench torque radius differs from m*g*r")

    actuation = _mapping(
        root_value["contact_and_actuation"], "contact_and_actuation"
    )
    _exact_keys(
        actuation,
        (
            "contact_count",
            "normal_force_cap_n",
            "independent_joint_names",
            "closing_joint_names",
            "closing_order_mode",
            "closing_order_permutations",
            "fixed_finger_1_first_physical_or_safety_evidence",
            "active_finger_first_contact_rule",
            "active_finger_pad_triangle_count",
            "active_finger_nonpad_triangle_count",
            "hand_drive_maximum_effort_nm",
            "contact_effort_rise_nm",
            "measured_effort_abort_nm",
            "required_closing_joint_effort_role",
            "hand_stiffness_nm_per_rad",
            "hand_damping_nm_s_per_rad",
            "pad_contract_15n_capacity_used",
            "urdf_100nm_effort_used",
        ),
        "contact_and_actuation",
    )
    contact_count = int(actuation["contact_count"])
    if contact_count != 3:
        raise AnalyticOuterMasterError("the frozen study requires exactly three contacts")
    normal_cap = _positive(
        actuation["normal_force_cap_n"], "normal force cap"
    )
    joints = _sequence_of_strings(
        actuation["independent_joint_names"], "independent joint names"
    )
    closing_joints = _sequence_of_strings(
        actuation["closing_joint_names"], "closing joint names"
    )
    if joints != EXPECTED_JOINTS or closing_joints != EXPECTED_CLOSING_JOINTS:
        raise AnalyticOuterMasterError("frozen independent/closing joint order changed")
    _exact_string(
        actuation["closing_order_mode"],
        "EXACTLY_ONE_OF_ALL_SIX_FINGER_PERMUTATIONS",
        "contact_and_actuation.closing_order_mode",
    )
    raw_closing_orders = actuation["closing_order_permutations"]
    if not isinstance(raw_closing_orders, Sequence) or isinstance(
        raw_closing_orders, (str, bytes)
    ):
        raise AnalyticOuterMasterError("closing orders must be a sequence")
    closing_orders = tuple(
        _sequence_of_strings(row, f"closing order {index}")
        for index, row in enumerate(raw_closing_orders)
    )
    if closing_orders != EXPECTED_CLOSING_ORDERS:
        raise AnalyticOuterMasterError(
            "closing-order decision must contain all six finger permutations"
        )
    _exact_bool(
        actuation["fixed_finger_1_first_physical_or_safety_evidence"],
        False,
        "fixed_finger_1_first_physical_or_safety_evidence",
    )
    active_finger_first_contact_rule = _exact_string(
        actuation["active_finger_first_contact_rule"],
        "COMPLETE_2442_TRIANGLE_PAD_BEFORE_SAME_FINGER_8912_TRIANGLE_NONPAD",
        "active_finger_first_contact_rule",
    )
    active_finger_pad_triangle_count = int(
        actuation["active_finger_pad_triangle_count"]
    )
    active_finger_nonpad_triangle_count = int(
        actuation["active_finger_nonpad_triangle_count"]
    )
    if active_finger_pad_triangle_count != EXPECTED_PAD_TRIANGLES:
        raise AnalyticOuterMasterError("active-finger pad triangle count changed")
    if active_finger_nonpad_triangle_count != 8912:
        raise AnalyticOuterMasterError("active-finger nonpad triangle count changed")
    hand_drive_cap = _positive(
        actuation["hand_drive_maximum_effort_nm"], "hand drive maximum effort"
    )
    contact_effort_rise = _positive(
        actuation["contact_effort_rise_nm"], "contact effort rise"
    )
    measured_effort_abort = _positive(
        actuation["measured_effort_abort_nm"], "measured effort abort"
    )
    if not contact_effort_rise < measured_effort_abort < hand_drive_cap:
        raise AnalyticOuterMasterError(
            "controller efforts must satisfy contact rise < measured abort < native drive cap"
        )
    required_effort_role = _exact_string(
        actuation["required_closing_joint_effort_role"],
        (
            "PRELOAD_POSITION_TARGET_ADVANCE_THRESHOLD_NOT_DRIVE_CAP_OR_"
            "REQUIRED_ACHIEVED_EFFORT"
        ),
        "required_closing_joint_effort_role",
    )
    stiffness = _positive(
        actuation["hand_stiffness_nm_per_rad"], "hand stiffness"
    )
    damping = _positive(
        actuation["hand_damping_nm_s_per_rad"], "hand damping"
    )
    _exact_bool(
        actuation["pad_contract_15n_capacity_used"],
        False,
        "pad_contract_15n_capacity_used",
    )
    _exact_bool(
        actuation["urdf_100nm_effort_used"],
        False,
        "urdf_100nm_effort_used",
    )

    outer = _mapping(root_value["outer_master"], "outer_master")
    _exact_keys(
        outer,
        (
            "objective",
            "pose_scenario",
            "wrench_scenarios",
            "friction_rule",
            "friction_outer_polygon_edges",
            "object_surface_mode",
            "pad_surface_mode",
            "shared_design_across_scenarios",
            "scenario_specific_contact_force_only",
            "omitted_for_outer_relaxation",
            "output_claim",
        ),
        "outer_master",
    )
    _exact_string(
        outer["objective"],
        "MINIMIZE_NEGATIVE_ACTUATOR_ENVELOPE_MARGIN",
        "outer_master.objective",
    )
    _exact_string(
        outer["pose_scenario"],
        "NOMINAL_ONLY_CONTINUOUS_POSE_SET_OMITTED_FOR_LOWER_BOUND",
        "outer_master.pose_scenario",
    )
    _exact_string(
        outer["wrench_scenarios"],
        "ZERO_AND_POSITIVE_NEGATIVE_SIX_AXIS_BOUNDARY",
        "outer_master.wrench_scenarios",
    )
    _exact_string(
        outer["friction_rule"],
        "CONTINUOUS_INTERVAL_REDUCED_EXACTLY_TO_LOWER_ENDPOINT",
        "outer_master.friction_rule",
    )
    edge_count = int(outer["friction_outer_polygon_edges"])
    if edge_count < 4 or edge_count % 2:
        raise AnalyticOuterMasterError("friction outer polygon needs an even edge count >= 4")
    _exact_string(
        outer["object_surface_mode"],
        (
            "ALL_NON_HARD_STEP_PARENT_FACE_AABB_UNION_AND_GLOBAL_AXIS_SUPPORT_"
            "OUTER_PER_FINGER"
        ),
        "outer_master.object_surface_mode",
    )
    _exact_string(
        outer["pad_surface_mode"],
        "ALL_2442_DIRECT_NAILFREE_PAD_TRIANGLES_EXACT_ONE_PER_FINGER",
        "outer_master.pad_surface_mode",
    )
    _exact_bool(
        outer["shared_design_across_scenarios"],
        True,
        "outer_master.shared_design_across_scenarios",
    )
    _exact_bool(
        outer["scenario_specific_contact_force_only"],
        True,
        "outer_master.scenario_specific_contact_force_only",
    )
    omissions = _sequence_of_strings(
        outer["omitted_for_outer_relaxation"], "outer model omissions"
    )
    if omissions != EXPECTED_OMISSIONS:
        raise AnalyticOuterMasterError("outer-model omission contract changed")
    _exact_string(
        outer["output_claim"],
        "FLOATING_POINT_SCIP_DUAL_BOUND_FOR_ENCODED_OUTER_MODEL_ONLY",
        "outer_master.output_claim",
    )

    contract = AnalyticEnvelopeContract(
        path=path,
        repository_root=root,
        step_contact_atlas_path=atlas_path,
        hand_contact_contract_path=hand_path,
        mass_kg=mass,
        center_of_mass_object_m=(com[0], com[1], com[2]),
        inertia_object_kg_m2=inertia,
        gravity_object_m_s2=(gravity[0], gravity[1], gravity[2]),
        visual_pose_empirical_hard_set_defined=(
            visual_pose_empirical_hard_set_defined
        ),
        visual_pose_empirical_hard_set_status=(
            visual_pose_empirical_hard_set_status
        ),
        visual_pose_empirical_hard_set_source=(
            visual_pose_empirical_hard_set_source
        ),
        visual_pose_measurement_protocol_executable=(
            visual_pose_measurement_protocol_executable
        ),
        visual_pose_measurement_protocol_status=(
            visual_pose_measurement_protocol_status
        ),
        visual_pose_observation_rule=visual_pose_observation_rule,
        visual_pose_estimate_truth_firewall=visual_pose_estimate_truth_firewall,
        visual_pose_residual_coordinates=visual_pose_residual_coordinates,
        visual_pose_residual_frame=visual_pose_residual_frame,
        visual_pose_residual_transform=visual_pose_residual_transform,
        visual_pose_orientation_branch_rule=visual_pose_orientation_branch_rule,
        visual_pose_failure_rule=visual_pose_failure_rule,
        visual_pose_empirical_support_rule=visual_pose_empirical_support_rule,
        visual_pose_miss_rule=visual_pose_miss_rule,
        visual_pose_empirical_support_claim=visual_pose_empirical_support_claim,
        visual_pose_truth_use=visual_pose_truth_use,
        visual_pose_research_set_defined=visual_pose_research_set_defined,
        visual_pose_research_set_frozen_before_robust_grasp_outcomes=(
            visual_pose_research_set_frozen_before_robust_grasp_outcomes
        ),
        visual_pose_research_set_role=visual_pose_research_set_role,
        visual_pose_research_set_source=visual_pose_research_set_source,
        visual_pose_research_set_coordinates=visual_pose_research_set_coordinates,
        visual_pose_research_set_lower=visual_pose_research_set_lower,
        visual_pose_research_set_upper=visual_pose_research_set_upper,
        visual_pose_research_set_guarantee_scope=(
            visual_pose_research_set_guarantee_scope
        ),
        visual_pose_research_set_transfer_limit=(
            visual_pose_research_set_transfer_limit
        ),
        stress_test_pose_translation_radius_m=stress_translation,
        stress_test_pose_rotation_radius_rad=stress_rotation,
        stress_test_pose_role=stress_role,
        friction_interval=(friction[0], friction[1]),
        wrench_force_radius_n=force_radius,
        wrench_torque_radius_nm=torque_radius,
        characteristic_radius_m=characteristic_radius,
        contact_count=contact_count,
        normal_force_cap_n=normal_cap,
        independent_joint_names=joints,
        closing_joint_names=closing_joints,
        closing_orders=closing_orders,
        active_finger_first_contact_rule=active_finger_first_contact_rule,
        active_finger_pad_triangle_count=active_finger_pad_triangle_count,
        active_finger_nonpad_triangle_count=active_finger_nonpad_triangle_count,
        hand_drive_maximum_effort_nm=hand_drive_cap,
        contact_effort_rise_nm=contact_effort_rise,
        measured_effort_abort_nm=measured_effort_abort,
        required_closing_joint_effort_role=required_effort_role,
        hand_stiffness_nm_per_rad=stiffness,
        hand_damping_nm_s_per_rad=damping,
        friction_outer_polygon_edges=edge_count,
        omitted_for_outer_relaxation=omissions,
    )
    if contract.force_only_gamma_upper_bound <= 0.0:
        raise AnalyticOuterMasterError("force-only gamma upper bound is not positive")
    return contract


@dataclass(frozen=True)
class WrenchScenario:
    name: str
    force_object_n: tuple[float, float, float]
    torque_object_nm: tuple[float, float, float]


def frozen_outer_scenarios(contract: AnalyticEnvelopeContract) -> tuple[WrenchScenario, ...]:
    rows = [WrenchScenario("zero", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    axes = ("x", "y", "z")
    for slot, axis in enumerate(axes):
        for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
            force = [0.0, 0.0, 0.0]
            force[slot] = sign * contract.wrench_force_radius_n
            rows.append(
                WrenchScenario(
                    f"force_{axis}_{sign_name}", tuple(force), (0.0, 0.0, 0.0)
                )
            )
    for slot, axis in enumerate(axes):
        for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
            torque = [0.0, 0.0, 0.0]
            torque[slot] = sign * contract.wrench_torque_radius_nm
            rows.append(
                WrenchScenario(
                    f"torque_{axis}_{sign_name}", (0.0, 0.0, 0.0), tuple(torque)
                )
            )
    return tuple(rows)


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitudes = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise AnalyticOuterMasterError("contact atlas has a degenerate triangle")
    return cross / magnitudes[:, None]


def _tangent_directions(normals: np.ndarray, edge_count: int) -> np.ndarray:
    result = np.empty((len(normals), edge_count, 3), dtype=np.float64)
    for index, normal in enumerate(normals):
        reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(normal @ reference)) > 0.8:
            reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        tangent_1 = np.cross(normal, reference)
        tangent_1 /= np.linalg.norm(tangent_1)
        tangent_2 = np.cross(normal, tangent_1)
        for edge in range(edge_count):
            angle = 2.0 * math.pi * edge / edge_count
            result[index, edge] = (
                math.cos(angle) * tangent_1 + math.sin(angle) * tangent_2
            )
    return result


@dataclass(frozen=True)
class PadSideForceEnvelope:
    """Per-triangle outer cones for unilateral force on the complete pad.

    A closed mesh triangle also contains its three edges and vertices.  The
    force cone associated with one selected triangle must therefore retain the
    normal cones of every face incident to those vertices.  Each row below is
    a circular cone enclosing that complete vertex-star normal cone, expanded
    once more by the circumscribed friction angle.  If no enclosing cone with
    half-angle below 90 degrees is available, the row is deliberately left
    unconstrained instead of deleting a possible edge/vertex contact.
    """

    center_local: np.ndarray
    cosine_half_angle: np.ndarray
    incident_normal_radius_rad: np.ndarray
    friction_half_angle_rad: float

    @property
    def constrained_triangle_count(self) -> int:
        return int(np.count_nonzero(self.cosine_half_angle > 0.0))

    @property
    def unconstrained_triangle_count(self) -> int:
        return int(len(self.cosine_half_angle) - self.constrained_triangle_count)


def _pad_side_force_envelope(
    pad: VerifiedPad,
    contract: AnalyticEnvelopeContract,
) -> PadSideForceEnvelope:
    """Enclose all pad face/edge/vertex Coulomb directions without pruning."""

    triangles = pad.points_local_m[pad.faces]
    normals = _triangle_normals(triangles)
    incident_faces: list[list[int]] = [[] for _ in range(pad.vertex_count)]
    for triangle_index, face in enumerate(pad.faces):
        for vertex in face:
            incident_faces[int(vertex)].append(triangle_index)

    centers = np.zeros((pad.triangle_count, 3), dtype=np.float64)
    cosines = np.zeros(pad.triangle_count, dtype=np.float64)
    radii = np.full(pad.triangle_count, math.pi, dtype=np.float64)
    mu_outer = contract.friction_minimum / math.cos(
        math.pi / contract.friction_outer_polygon_edges
    )
    friction_angle = math.atan(mu_outer)
    angle_margin = 256.0 * np.finfo(np.float64).eps

    for triangle_index, face in enumerate(pad.faces):
        neighbours = sorted(
            {
                incident
                for vertex in face
                for incident in incident_faces[int(vertex)]
            }
        )
        neighbour_normals = normals[np.asarray(neighbours, dtype=np.int64)]
        center = np.sum(neighbour_normals, axis=0)
        center_norm = float(np.linalg.norm(center))
        if not math.isfinite(center_norm) or center_norm <= angle_margin:
            continue
        center /= center_norm
        radius = float(
            np.max(
                np.arccos(
                    np.clip(neighbour_normals @ center, -1.0, 1.0)
                )
            )
        )
        radius = math.nextafter(radius + angle_margin, math.inf)
        radii[triangle_index] = radius
        half_angle = radius + friction_angle + angle_margin
        if half_angle >= 0.5 * math.pi:
            continue
        cosine = math.nextafter(math.cos(half_angle), -math.inf)
        if not math.isfinite(cosine) or cosine <= 0.0:
            continue
        centers[triangle_index] = center
        cosines[triangle_index] = cosine

    centers.setflags(write=False)
    cosines.setflags(write=False)
    radii.setflags(write=False)
    return PadSideForceEnvelope(
        center_local=centers,
        cosine_half_angle=cosines,
        incident_normal_radius_rad=radii,
        friction_half_angle_rad=friction_angle,
    )


@dataclass(frozen=True)
class DirectionalForceGammaBound:
    gamma_upper_bound: float
    limiting_scenario: str
    single_contact_support_n: float
    contact_force_axis_supports: tuple[
        tuple[str, tuple[float, float, float], float], ...
    ]


def _directional_force_gamma_bound(
    contract: AnalyticEnvelopeContract,
    normals: np.ndarray,
    tangents: np.ndarray,
    scenarios: Sequence[WrenchScenario],
) -> DirectionalForceGammaBound:
    """Bound gamma with support functions of the encoded outer friction cones."""

    edge_count = contract.friction_outer_polygon_edges
    vertex_radius = contract.friction_minimum / math.cos(math.pi / edge_count)
    polygon_vertices = np.empty_like(tangents)
    for edge in range(edge_count):
        direction = tangents[:, edge] + tangents[:, (edge + 1) % edge_count]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        polygon_vertices[:, edge] = vertex_radius * direction

    gravity_force = contract.mass_kg * np.asarray(
        contract.gravity_object_m_s2, dtype=np.float64
    )
    best: DirectionalForceGammaBound | None = None
    axis_supports: list[tuple[str, tuple[float, float, float], float]] = []
    for scenario in scenarios:
        force = np.asarray(scenario.force_object_n, dtype=np.float64)
        magnitude = float(np.linalg.norm(force))
        if magnitude == 0.0:
            continue
        unit = force / magnitude
        contact_projection = -unit
        tangent_support = np.max(
            polygon_vertices @ contact_projection, axis=1
        )
        per_triangle_support = contract.normal_force_cap_n * np.maximum(
            0.0, -normals @ contact_projection + tangent_support
        )
        raw_support = float(np.max(per_triangle_support))
        support_margin = 64.0 * np.finfo(np.float64).eps * max(1.0, raw_support)
        single_contact_support = math.nextafter(
            raw_support + support_margin, math.inf
        )
        axis_supports.append(
            (
                scenario.name,
                tuple(float(value) for value in contact_projection),
                single_contact_support,
            )
        )
        gamma_bound = (
            contract.contact_count * single_contact_support
            - float(unit @ gravity_force)
        ) / magnitude
        if gamma_bound < 0.0:
            gamma_bound = 0.0
        row = DirectionalForceGammaBound(
            gamma_upper_bound=gamma_bound,
            limiting_scenario=scenario.name,
            single_contact_support_n=single_contact_support,
            contact_force_axis_supports=(),
        )
        if best is None or row.gamma_upper_bound < best.gamma_upper_bound:
            best = row
    if best is None or not math.isfinite(best.gamma_upper_bound):
        raise AnalyticOuterMasterError("no finite force-direction gamma bound exists")
    return DirectionalForceGammaBound(
        gamma_upper_bound=best.gamma_upper_bound,
        limiting_scenario=best.limiting_scenario,
        single_contact_support_n=best.single_contact_support_n,
        contact_force_axis_supports=tuple(axis_supports),
    )


def _matmul(left: Sequence[Sequence[Any]], right: Sequence[Sequence[Any]]) -> list[list[Any]]:
    if not left or not right or len(left[0]) != len(right):
        raise AnalyticOuterMasterError("symbolic matrix shapes do not align")
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
         for column in range(len(right[0]))]
        for row in range(len(left))
    ]


def _numeric_matrix(value: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in np.asarray(value)]


def _axis_rotation_expression(axis: Sequence[float], angle: Any, sin: Any, cos: Any) -> list[list[Any]]:
    x, y, z = (float(item) for item in axis)
    skew = (
        (0.0, -z, y),
        (z, 0.0, -x),
        (-y, x, 0.0),
    )
    skew_array = np.asarray(skew, dtype=np.float64)
    skew_squared = skew_array @ skew_array
    sine = sin(angle)
    cosine = cos(angle)
    rotation = []
    for row in range(3):
        values = []
        for column in range(3):
            identity = 1.0 if row == column else 0.0
            values.append(
                identity
                + sine * float(skew_array[row, column])
                + (1.0 - cosine) * float(skew_squared[row, column])
            )
        rotation.append(values)
    return rotation


def _motion_transform_expression(joint: Any, position: Any, sin: Any, cos: Any) -> list[list[Any]]:
    transform: list[list[Any]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if joint.joint_type in ("revolute", "continuous"):
        rotation = _axis_rotation_expression(joint.axis, position, sin, cos)
        for row in range(3):
            for column in range(3):
                transform[row][column] = rotation[row][column]
    elif joint.joint_type == "prismatic":
        for row in range(3):
            transform[row][3] = float(joint.axis[row]) * position
    elif joint.joint_type != "fixed":
        raise AnalyticOuterMasterError(f"unsupported joint type {joint.joint_type}")
    return transform


def _resolved_joint_expression(
    hand: ThreeFingerHandModel,
    joint_name: str,
    independent_positions: Mapping[str, Any],
    cache: dict[str, Any],
) -> Any:
    if joint_name in cache:
        return cache[joint_name]
    joint = hand.joints[joint_name]
    if joint.mimic is None:
        result = independent_positions[joint_name]
    else:
        result = (
            joint.mimic.multiplier
            * _resolved_joint_expression(
                hand, joint.mimic.source_joint, independent_positions, cache
            )
            + joint.mimic.offset
        )
    cache[joint_name] = result
    return result


def _affine_source(
    hand: ThreeFingerHandModel,
    joint_name: str,
    cache: dict[str, tuple[str, float]],
) -> tuple[str, float]:
    if joint_name in cache:
        return cache[joint_name]
    joint = hand.joints[joint_name]
    if joint.mimic is None:
        result = (joint_name, 1.0)
    else:
        source, multiplier = _affine_source(hand, joint.mimic.source_joint, cache)
        result = (source, multiplier * joint.mimic.multiplier)
    cache[joint_name] = result
    return result


def _symbolic_forward_kinematics(
    hand: ThreeFingerHandModel,
    independent_positions: Mapping[str, Any],
    sin: Any,
    cos: Any,
) -> Mapping[str, list[list[Any]]]:
    identity: list[list[Any]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transforms: dict[str, list[list[Any]]] = {hand.base_link: identity}
    position_cache: dict[str, Any] = {}
    for joint_name in hand.joint_order:
        joint = hand.joints[joint_name]
        position = (
            0.0
            if not joint.movable
            else _resolved_joint_expression(
                hand, joint_name, independent_positions, position_cache
            )
        )
        origin = _numeric_matrix(joint.origin_transform())
        motion = _motion_transform_expression(joint, position, sin, cos)
        transforms[joint.child_link] = _matmul(
            _matmul(transforms[joint.parent_link], origin), motion
        )
    return transforms


def _ancestor_joint_names(hand: ThreeFingerHandModel, link_name: str) -> tuple[str, ...]:
    by_child = {joint.child_link: name for name, joint in hand.joints.items()}
    names: list[str] = []
    cursor = link_name
    while cursor != hand.base_link:
        joint_name = by_child.get(cursor)
        if joint_name is None:
            raise AnalyticOuterMasterError(f"link {link_name} is disconnected")
        names.append(joint_name)
        cursor = hand.joints[joint_name].parent_link
    names.reverse()
    return tuple(names)


def _hand_contact_radius_bound(
    hand: ThreeFingerHandModel, hand_contract: CARTSHandContract
) -> float:
    largest = 0.0
    for pad in hand_contract.pads:
        path = _ancestor_joint_names(hand, pad.link_name)
        origin_radius = sum(
            float(np.linalg.norm(hand.joints[name].origin_xyz_m)) for name in path
        )
        point_radius = float(np.max(np.linalg.norm(pad.points_local_m, axis=1)))
        largest = max(largest, origin_radius + point_radius)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, largest)
    return largest + tolerance


def _quaternion_rotation(quaternion: Sequence[Any]) -> list[list[Any]]:
    w, x, y, z = quaternion
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _cross(left: Sequence[Any], right: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add_exact_triangle_surface_point(
    model: Any,
    quicksum: Any,
    triangles: np.ndarray,
    *,
    prefix: str,
) -> tuple[list[Any], list[Any]]:
    lower = np.min(triangles.reshape(-1, 3), axis=0)
    upper = np.max(triangles.reshape(-1, 3), axis=0)
    point = [
        model.addVar(lb=float(lower[axis]), ub=float(upper[axis]), name=f"{prefix}_p{axis}")
        for axis in range(3)
    ]
    selectors: list[Any] = []
    weights: list[tuple[Any, Any, Any]] = []
    for triangle_index in range(len(triangles)):
        selector = model.addVar(vtype="B", name=f"{prefix}_z{triangle_index}")
        bary = tuple(
            model.addVar(lb=0.0, ub=1.0, name=f"{prefix}_b{triangle_index}_{vertex}")
            for vertex in range(3)
        )
        model.addCons(
            quicksum(bary) == selector,
            name=f"{prefix}_simplex{triangle_index}",
        )
        selectors.append(selector)
        weights.append(bary)
    model.addCons(quicksum(selectors) == 1.0, name=f"{prefix}_exact_one")
    for axis in range(3):
        model.addCons(
            point[axis]
            == quicksum(
                weights[triangle][vertex] * float(triangles[triangle, vertex, axis])
                for triangle in range(len(triangles))
                for vertex in range(3)
            ),
            name=f"{prefix}_point_axis{axis}",
        )
    return point, selectors


def _triangle_union_aabb(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact coordinate box of a frozen closed triangle union."""

    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise AnalyticOuterMasterError("triangle union must have shape (N,3,3)")
    if not len(values) or not np.all(np.isfinite(values)):
        raise AnalyticOuterMasterError("triangle union is empty or non-finite")
    lower = np.min(values, axis=(0, 1))
    upper = np.max(values, axis=(0, 1))
    if np.any(lower > upper):
        raise AnalyticOuterMasterError("triangle-union AABB is invalid")
    return lower, upper


@dataclass(frozen=True)
class _ParentFaceAABBVariables:
    point: tuple[Any, Any, Any]
    parent_face_indices: tuple[int, ...]
    selectors: tuple[Any, ...]
    lower_m: tuple[tuple[float, float, float], ...]
    upper_m: tuple[tuple[float, float, float], ...]


def _add_parent_face_aabb_union_point(
    model: Any,
    quicksum: Any,
    atlas: StepContactAtlas,
    *,
    prefix: str,
) -> _ParentFaceAABBVariables:
    """Add one point in the exact-one union of all parent-face AABBs.

    Every frozen non-hard triangle is contained in the AABB of its own STEP
    parent face.  The disjunction is therefore a tighter outer relaxation of
    the complete atlas; parent identifiers are computational partitions, not
    contact permissions or physical barriers.
    """

    parent_face_indices = tuple(
        int(value) for value in np.unique(atlas.parent_face_index)
    )
    if not parent_face_indices:
        raise AnalyticOuterMasterError("object atlas has no parent-face group")
    if parent_face_indices != tuple(atlas.contract.allowed_parent_faces):
        raise AnalyticOuterMasterError(
            "parent-face AABB modes do not cover every frozen non-hard parent"
        )
    global_lower, global_upper = _triangle_union_aabb(atlas.triangles_m)
    global_lower = np.nextafter(global_lower, -np.inf)
    global_upper = np.nextafter(global_upper, np.inf)
    lower_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []
    selectors: list[Any] = []
    for parent_face in parent_face_indices:
        indices = np.flatnonzero(atlas.parent_face_index == parent_face)
        if not len(indices):
            raise AnalyticOuterMasterError("parent-face AABB group is empty")
        lower, upper = _triangle_union_aabb(atlas.triangles_m[indices])
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        lower_rows.append(lower)
        upper_rows.append(upper)
        selectors.append(
            model.addVar(vtype="B", name=f"{prefix}_parent{parent_face}")
        )
    model.addCons(quicksum(selectors) == 1.0, name=f"{prefix}_exact_one_parent")

    point = tuple(
        model.addVar(
            lb=float(global_lower[axis]),
            ub=float(global_upper[axis]),
            name=f"{prefix}_p{axis}",
        )
        for axis in range(3)
    )
    for axis in range(3):
        model.addCons(
            point[axis]
            >= quicksum(
                selectors[row] * float(lower_rows[row][axis])
                for row in range(len(parent_face_indices))
            ),
            name=f"{prefix}_parent_aabb_lower{axis}",
        )
        model.addCons(
            point[axis]
            <= quicksum(
                selectors[row] * float(upper_rows[row][axis])
                for row in range(len(parent_face_indices))
            ),
            name=f"{prefix}_parent_aabb_upper{axis}",
        )
    return _ParentFaceAABBVariables(
        point=point,
        parent_face_indices=parent_face_indices,
        selectors=tuple(selectors),
        lower_m=tuple(tuple(float(value) for value in row) for row in lower_rows),
        upper_m=tuple(tuple(float(value) for value in row) for row in upper_rows),
    )


@dataclass(frozen=True)
class _SurfaceUnionVariables:
    point: tuple[Any, Any, Any]
    selectors: tuple[Any, ...]
    barycentric: tuple[tuple[Any, Any, Any], ...]


def _add_closed_triangle_union(
    model: Any,
    quicksum: Any,
    triangles: np.ndarray,
    *,
    prefix: str,
) -> _SurfaceUnionVariables:
    """Exact-one closed triangle union used by the axial response outer model."""

    values = np.asarray(triangles, dtype=np.float64)
    lower = np.min(values.reshape(-1, 3), axis=0)
    upper = np.max(values.reshape(-1, 3), axis=0)
    point = tuple(
        model.addVar(
            lb=float(lower[axis]), ub=float(upper[axis]), name=f"{prefix}_p{axis}"
        )
        for axis in range(3)
    )
    selectors: list[Any] = []
    weights: list[tuple[Any, Any, Any]] = []
    for triangle_index in range(len(values)):
        selector = model.addVar(vtype="B", name=f"{prefix}_z{triangle_index}")
        barycentric = tuple(
            model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"{prefix}_b{triangle_index}_{vertex}",
            )
            for vertex in range(3)
        )
        model.addCons(
            quicksum(barycentric) == selector,
            name=f"{prefix}_simplex{triangle_index}",
        )
        selectors.append(selector)
        weights.append(barycentric)
    model.addCons(quicksum(selectors) == 1.0, name=f"{prefix}_exact_one")
    for axis in range(3):
        model.addCons(
            point[axis]
            == quicksum(
                weights[triangle][vertex] * float(values[triangle, vertex, axis])
                for triangle in range(len(values))
                for vertex in range(3)
            ),
            name=f"{prefix}_point_axis{axis}",
        )
    return _SurfaceUnionVariables(point, tuple(selectors), tuple(weights))


def _add_exact_parent_surface_union(
    model: Any,
    quicksum: Any,
    sin: Any,
    cos: Any,
    atlas: StepContactAtlas,
    *,
    prefix: str,
) -> _SurfaceUnionVariables:
    """Closed UV triangle union evaluated on complete analytic carriers only."""

    analytic_faces = {row.face_index for row in atlas.parent_surfaces}
    encoded_faces = {int(value) for value in np.unique(atlas.parent_face_index)}
    unsupported_faces = encoded_faces - analytic_faces
    if unsupported_faces:
        unsupported_triangles = int(
            np.count_nonzero(
                np.isin(
                    atlas.parent_face_index,
                    np.asarray(sorted(unsupported_faces), dtype=np.int64),
                )
            )
        )
        raise AnalyticOuterMasterError(
            "analytic parent-surface union does not cover the corrected outer "
            f"domain: {len(unsupported_faces)} parent faces and "
            f"{unsupported_triangles} triangles lack a plane/cylinder carrier"
        )

    selectors: list[Any] = []
    weights: list[tuple[Any, Any, Any]] = []
    for triangle_index in range(atlas.triangle_count):
        selector = model.addVar(vtype="B", name=f"{prefix}_z{triangle_index}")
        barycentric = tuple(
            model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"{prefix}_b{triangle_index}_{vertex}",
            )
            for vertex in range(3)
        )
        model.addCons(
            quicksum(barycentric) == selector,
            name=f"{prefix}_simplex{triangle_index}",
        )
        selectors.append(selector)
        weights.append(barycentric)
    model.addCons(quicksum(selectors) == 1.0, name=f"{prefix}_exact_one")

    contributions: list[tuple[Any, Any, Any]] = []
    for parent_row, parent in enumerate(atlas.parent_surfaces):
        indices = np.flatnonzero(atlas.parent_face_index == parent.face_index)
        parent_selector = model.addVar(vtype="B", name=f"{prefix}_parent{parent.face_index}")
        model.addCons(
            parent_selector == quicksum(selectors[int(index)] for index in indices),
            name=f"{prefix}_parent{parent.face_index}_active",
        )
        u_expression = quicksum(
            weights[int(index)][vertex]
            * float(atlas.triangle_uv[int(index), vertex, 0])
            for index in indices
            for vertex in range(3)
        )
        v_expression = quicksum(
            weights[int(index)][vertex]
            * float(atlas.triangle_uv[int(index), vertex, 1])
            for index in indices
            for vertex in range(3)
        )
        if parent.kind == "plane":
            contribution = tuple(
                parent_selector * float(parent.origin_m[axis])
                + parent.uv_length_scale_m
                * (
                    u_expression * float(parent.x_direction[axis])
                    + v_expression * float(parent.y_direction[axis])
                )
                for axis in range(3)
            )
        else:
            assert parent.radius_m is not None
            cosine = model.addVar(lb=-1.0, ub=1.0, name=f"{prefix}_c{parent_row}")
            sine = model.addVar(lb=-1.0, ub=1.0, name=f"{prefix}_s{parent_row}")
            model.addCons(cosine <= parent_selector, name=f"{prefix}_c{parent_row}_ub")
            model.addCons(-cosine <= parent_selector, name=f"{prefix}_c{parent_row}_lb")
            model.addCons(sine <= parent_selector, name=f"{prefix}_s{parent_row}_ub")
            model.addCons(-sine <= parent_selector, name=f"{prefix}_s{parent_row}_lb")
            # Inactive parents have u=0, cos(u)=1 and c=0; M=2 safely
            # deactivates the equality.  An active parent enforces it exactly.
            model.addCons(
                cosine - cos(u_expression) <= 2.0 * (1.0 - parent_selector),
                name=f"{prefix}_c{parent_row}_pos",
            )
            model.addCons(
                cos(u_expression) - cosine <= 2.0 * (1.0 - parent_selector),
                name=f"{prefix}_c{parent_row}_neg",
            )
            model.addCons(
                sine - sin(u_expression) <= 2.0 * (1.0 - parent_selector),
                name=f"{prefix}_s{parent_row}_pos",
            )
            model.addCons(
                sin(u_expression) - sine <= 2.0 * (1.0 - parent_selector),
                name=f"{prefix}_s{parent_row}_neg",
            )
            contribution = tuple(
                parent_selector * float(parent.origin_m[axis])
                + parent.radius_m
                * (
                    cosine * float(parent.x_direction[axis])
                    + sine * float(parent.y_direction[axis])
                )
                + parent.uv_length_scale_m
                * v_expression
                * float(parent.axis_direction[axis])
                for axis in range(3)
            )
        contributions.append(contribution)

    object_radius = max(
        float(np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))),
        max(
            float(np.linalg.norm(parent.origin_m)) + float(parent.radius_m or 0.0)
            for parent in atlas.parent_surfaces
        ),
    )
    point = tuple(
        model.addVar(
            lb=-object_radius, ub=object_radius, name=f"{prefix}_p{axis}"
        )
        for axis in range(3)
    )
    for axis in range(3):
        model.addCons(
            point[axis] == quicksum(row[axis] for row in contributions),
            name=f"{prefix}_point_axis{axis}",
        )
    return _SurfaceUnionVariables(point, tuple(selectors), tuple(weights))


@dataclass
class OuterMasterBundle:
    model: Any
    contract: AnalyticEnvelopeContract
    atlas: StepContactAtlas
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    scenarios: tuple[WrenchScenario, ...]
    gamma: Any
    quaternion: tuple[Any, Any, Any, Any]
    translation_hc: tuple[Any, Any, Any]
    q_pre: Mapping[str, Any]
    q_contact: Mapping[str, Any]
    q_goal: Mapping[str, Any]
    closing_order_selectors: Mapping[tuple[str, str, str], Any]
    object_contact_points: tuple[tuple[Any, Any, Any], ...]
    pad_contact_points: tuple[tuple[Any, Any, Any], ...]
    object_parent_face_aabb_variables: tuple[
        _ParentFaceAABBVariables,
        _ParentFaceAABBVariables,
        _ParentFaceAABBVariables,
    ]
    pad_triangle_selectors: tuple[
        tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]
    ]
    object_aabb_lower_m: tuple[float, float, float]
    object_aabb_upper_m: tuple[float, float, float]
    translation_bound_m: float
    force_ball_gamma_upper_bound: float
    directional_force_bound: DirectionalForceGammaBound
    pad_side_force_envelopes: tuple[
        PadSideForceEnvelope, PadSideForceEnvelope, PadSideForceEnvelope
    ]


@dataclass
class AxialEndpointOuterBundle:
    model: Any
    contract: AnalyticEnvelopeContract
    atlas: StepContactAtlas
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    axial_errors_m: tuple[float, float]
    quaternion: tuple[Any, Any, Any, Any]
    translation_hc: tuple[Any, Any, Any]
    q_pre: Mapping[str, Any]
    q_goal: Mapping[str, Any]
    q_contact_by_scenario: tuple[Mapping[str, Any], Mapping[str, Any]]
    object_surfaces: tuple[
        tuple[_SurfaceUnionVariables, _SurfaceUnionVariables, _SurfaceUnionVariables],
        tuple[_SurfaceUnionVariables, _SurfaceUnionVariables, _SurfaceUnionVariables],
    ]
    pad_surfaces: tuple[
        tuple[_SurfaceUnionVariables, _SurfaceUnionVariables, _SurfaceUnionVariables],
        tuple[_SurfaceUnionVariables, _SurfaceUnionVariables, _SurfaceUnionVariables],
    ]
    translation_bound_m: float


def build_axial_endpoint_outer_master(
    contract: AnalyticEnvelopeContract,
) -> AxialEndpointOuterBundle:
    """Build the zero-relative-motion mode-seed screen at axial U endpoints."""

    if not contract.visual_pose_research_set_defined:
        raise AnalyticOuterMasterError(
            "the visual-pose research set is undefined; the old +/-10 mm endpoints "
            "are stress-test values and cannot define this optimization screen"
        )

    try:
        from pyscipopt import Model, cos, quicksum, sin
    except ImportError as error:
        raise AnalyticOuterMasterError("PySCIPOpt is unavailable") from error

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path, repository_root=contract.repository_root
    )
    hand = hand_contract.build_hand_model()
    if any(pad.triangle_count != EXPECTED_PAD_TRIANGLES for pad in hand_contract.pads):
        raise AnalyticOuterMasterError("complete-pad mode count changed")
    if not atlas.parent_surfaces:
        raise AnalyticOuterMasterError(
            "zero-relative-motion screen has no proven analytic parent carrier"
        )
    if tuple(hand.independent_joint_names) != contract.independent_joint_names:
        raise AnalyticOuterMasterError("URDF independent joint order differs from contract")
    axial_errors = (
        float(contract.visual_pose_research_set_lower[2]),
        float(contract.visual_pose_research_set_upper[2]),
    )

    model = Model("te_zero_relative_motion_two_endpoint_intersection_screen_v2")
    joint_lower, joint_upper = hand.joint_limit_vectors()
    q_pre = {
        name: model.addVar(
            lb=float(joint_lower[index]),
            ub=float(joint_upper[index]),
            name=f"qpre_{name}",
        )
        for index, name in enumerate(hand.independent_joint_names)
    }
    q_goal = {
        name: model.addVar(
            lb=float(joint_lower[index]),
            ub=float(joint_upper[index]),
            name=f"qgoal_{name}",
        )
        for index, name in enumerate(hand.independent_joint_names)
    }
    model.addCons(
        q_pre["f1j1"] == q_goal["f1j1"],
        name="shared_palm_fixed_during_closure",
    )
    q_contact_rows: list[Mapping[str, Any]] = []
    for scenario_index in range(2):
        row = {
            name: model.addVar(
                lb=float(joint_lower[index]),
                ub=float(joint_upper[index]),
                name=f"e{scenario_index}_qcontact_{name}",
            )
            for index, name in enumerate(hand.independent_joint_names)
        }
        model.addCons(
            row["f1j1"] == q_pre["f1j1"],
            name=f"e{scenario_index}_palm_contact_fixed",
        )
        for name in contract.closing_joint_names:
            model.addCons(
                q_pre[name] <= row[name],
                name=f"e{scenario_index}_{name}_pre_before_contact",
            )
            model.addCons(
                row[name] <= q_goal[name],
                name=f"e{scenario_index}_{name}_contact_before_goal",
            )
        q_contact_rows.append(row)

    quaternion = (
        model.addVar(lb=0.0, ub=1.0, name="quat_w"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_x"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_y"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_z"),
    )
    model.addCons(
        quicksum(value * value for value in quaternion) == 1.0,
        name="unit_quaternion",
    )
    rotation_expression = _quaternion_rotation(quaternion)
    rotation_hc = [
        [
            model.addVar(lb=-1.0, ub=1.0, name=f"Rhc_{row}{column}")
            for column in range(3)
        ]
        for row in range(3)
    ]
    for row in range(3):
        for column in range(3):
            model.addCons(
                rotation_hc[row][column] == rotation_expression[row][column],
                name=f"quaternion_rotation_{row}{column}",
            )

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = max(
        float(np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))),
        max(
            float(np.linalg.norm(parent.origin_m)) + float(parent.radius_m or 0.0)
            for parent in atlas.parent_surfaces
        ),
    )
    maximum_translation_error = float(
        np.linalg.norm(
            np.maximum(
                np.abs(np.asarray(contract.visual_pose_research_set_lower[:3])),
                np.abs(np.asarray(contract.visual_pose_research_set_upper[:3])),
            )
        )
    )
    translation_bound = hand_radius + object_radius + maximum_translation_error
    translation_hc = tuple(
        model.addVar(
            lb=-translation_bound,
            ub=translation_bound,
            name=f"thc_{axis}",
        )
        for axis in range(3)
    )

    object_rows: list[tuple[_SurfaceUnionVariables, ...]] = []
    pad_rows: list[tuple[_SurfaceUnionVariables, ...]] = []
    for scenario_index, axial_error in enumerate(axial_errors):
        transforms = _symbolic_forward_kinematics(
            hand, q_contact_rows[scenario_index], sin, cos
        )
        object_surfaces: list[_SurfaceUnionVariables] = []
        pad_surfaces: list[_SurfaceUnionVariables] = []
        for finger_index, pad in enumerate(hand_contract.pads):
            prefix = f"e{scenario_index}_finger{finger_index}"
            object_surface = _add_exact_parent_surface_union(
                model,
                quicksum,
                sin,
                cos,
                atlas,
                prefix=f"{prefix}_object",
            )
            pad_surface = _add_closed_triangle_union(
                model,
                quicksum,
                pad.points_local_m[pad.faces],
                prefix=f"{prefix}_pad",
            )
            object_surfaces.append(object_surface)
            pad_surfaces.append(pad_surface)
            link_transform = transforms[pad.link_name]
            hand_point = tuple(
                model.addVar(
                    lb=-hand_radius,
                    ub=hand_radius,
                    name=f"{prefix}_ph{axis}",
                )
                for axis in range(3)
            )
            for axis in range(3):
                model.addCons(
                    hand_point[axis]
                    == sum(
                        link_transform[axis][column] * pad_surface.point[column]
                        for column in range(3)
                    )
                    + link_transform[axis][3],
                    name=f"{prefix}_fk_contact_{axis}",
                )
                model.addCons(
                    hand_point[axis]
                    == sum(
                        rotation_hc[axis][column]
                        * (
                            object_surface.point[column]
                            + (axial_error if column == 2 else 0.0)
                        )
                        for column in range(3)
                    )
                    + translation_hc[axis],
                    name=f"{prefix}_shared_design_contact_{axis}",
                )
        object_rows.append(tuple(object_surfaces))
        pad_rows.append(tuple(pad_surfaces))

    model.setObjective(0.0, "minimize")
    return AxialEndpointOuterBundle(
        model=model,
        contract=contract,
        atlas=atlas,
        hand_contract=hand_contract,
        hand_model=hand,
        axial_errors_m=axial_errors,
        quaternion=quaternion,
        translation_hc=translation_hc,
        q_pre=q_pre,
        q_goal=q_goal,
        q_contact_by_scenario=(q_contact_rows[0], q_contact_rows[1]),
        object_surfaces=(
            (object_rows[0][0], object_rows[0][1], object_rows[0][2]),
            (object_rows[1][0], object_rows[1][1], object_rows[1][2]),
        ),
        pad_surfaces=(
            (pad_rows[0][0], pad_rows[0][1], pad_rows[0][2]),
            (pad_rows[1][0], pad_rows[1][1], pad_rows[1][2]),
        ),
        translation_bound_m=translation_bound,
    )


def build_outer_master(
    contract: AnalyticEnvelopeContract,
) -> OuterMasterBundle:
    try:
        from pyscipopt import Model, cos, quicksum, sin
    except ImportError as error:
        raise AnalyticOuterMasterError("PySCIPOpt is unavailable") from error

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path, repository_root=contract.repository_root
    )
    if hand_contract.hardware_authorized:
        raise AnalyticOuterMasterError("hand contract unexpectedly authorizes hardware")
    hand = hand_contract.build_hand_model()
    if tuple(hand.independent_joint_names) != contract.independent_joint_names:
        raise AnalyticOuterMasterError("URDF independent joint order differs from contract")
    if len(hand_contract.pads) != contract.contact_count:
        raise AnalyticOuterMasterError("hand contract does not contain three pads")
    for pad in hand_contract.pads:
        if pad.triangle_count != EXPECTED_PAD_TRIANGLES:
            raise AnalyticOuterMasterError(f"{pad.name} triangle count changed")
        if pad.coordinate_frame != pad.link_name:
            raise AnalyticOuterMasterError(f"{pad.name} vertices are not link-local")
    pad_side_force_envelopes = tuple(
        _pad_side_force_envelope(pad, contract) for pad in hand_contract.pads
    )

    scenarios = frozen_outer_scenarios(contract)
    if len(scenarios) != 13:
        raise AnalyticOuterMasterError("frozen outer scenario count changed")
    object_aabb_lower, object_aabb_upper = _triangle_union_aabb(
        atlas.triangles_m
    )
    force_ball_gamma_upper_bound = contract.force_only_gamma_upper_bound
    normals = _triangle_normals(atlas.triangles_m)
    tangents = _tangent_directions(
        normals, contract.friction_outer_polygon_edges
    )
    directional_force_bound = _directional_force_gamma_bound(
        contract, normals, tangents, scenarios
    )

    model = Model("te_continuous_grasp_actuator_envelope_compact_root_v5")
    gamma = model.addVar(
        lb=0.0,
        ub=directional_force_bound.gamma_upper_bound,
        name="gamma_ae",
    )

    joint_lower, joint_upper = hand.joint_limit_vectors()
    q_pre = {
        name: model.addVar(lb=float(joint_lower[index]), ub=float(joint_upper[index]), name=f"qpre_{name}")
        for index, name in enumerate(hand.independent_joint_names)
    }
    q_contact = {
        name: model.addVar(lb=float(joint_lower[index]), ub=float(joint_upper[index]), name=f"qcontact_{name}")
        for index, name in enumerate(hand.independent_joint_names)
    }
    q_goal = {
        name: model.addVar(lb=float(joint_lower[index]), ub=float(joint_upper[index]), name=f"qgoal_{name}")
        for index, name in enumerate(hand.independent_joint_names)
    }
    model.addCons(q_pre["f1j1"] == q_goal["f1j1"], name="palm_target_fixed_during_closure")
    for name in contract.closing_joint_names:
        model.addCons(q_pre[name] <= q_contact[name], name=f"{name}_pre_before_contact")
        model.addCons(q_contact[name] <= q_goal[name], name=f"{name}_contact_before_goal")
    closing_order_selectors = {
        order: model.addVar(
            vtype="B",
            name="closure_order_" + "_then_".join(order),
        )
        for order in contract.closing_orders
    }
    model.addCons(
        quicksum(closing_order_selectors.values()) == 1.0,
        name="exactly_one_of_all_six_closing_orders",
    )
    quaternion = (
        model.addVar(lb=0.0, ub=1.0, name="quat_w"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_x"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_y"),
        model.addVar(lb=-1.0, ub=1.0, name="quat_z"),
    )
    model.addCons(
        quicksum(value * value for value in quaternion) == 1.0,
        name="unit_quaternion",
    )
    rotation_hc_expression = _quaternion_rotation(quaternion)
    rotation_hc = [
        [model.addVar(lb=-1.0, ub=1.0, name=f"Rhc_{row}{column}") for column in range(3)]
        for row in range(3)
    ]
    for row in range(3):
        for column in range(3):
            model.addCons(
                rotation_hc[row][column] == rotation_hc_expression[row][column],
                name=f"quaternion_rotation_{row}{column}",
            )

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = float(
        np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))
    )
    translation_bound = hand_radius + object_radius
    translation_hc = tuple(
        model.addVar(lb=-translation_bound, ub=translation_bound, name=f"thc_{axis}")
        for axis in range(3)
    )

    transforms = _symbolic_forward_kinematics(hand, q_contact, sin, cos)
    object_points: list[tuple[Any, Any, Any]] = []
    pad_points: list[tuple[Any, Any, Any]] = []
    object_parent_face_aabb_rows: list[_ParentFaceAABBVariables] = []
    pad_selector_rows: list[tuple[Any, ...]] = []
    hand_points: list[tuple[Any, Any, Any]] = []
    linear_jacobians: list[list[list[Any]]] = []
    pad_force_centers_hand: list[tuple[Any, Any, Any]] = []
    pad_force_cosines: list[Any] = []
    joint_column = {name: index for index, name in enumerate(hand.independent_joint_names)}
    affine_cache: dict[str, tuple[str, float]] = {}

    for finger_index, pad in enumerate(hand_contract.pads):
        object_parent_face_aabb = _add_parent_face_aabb_union_point(
            model,
            quicksum,
            atlas,
            prefix=f"finger{finger_index}_object",
        )
        object_point = object_parent_face_aabb.point
        pad_triangles = pad.points_local_m[pad.faces]
        pad_point, pad_selectors = _add_exact_triangle_surface_point(
            model,
            quicksum,
            pad_triangles,
            prefix=f"finger{finger_index}_pad",
        )
        object_points.append(tuple(object_point))
        pad_points.append(tuple(pad_point))
        object_parent_face_aabb_rows.append(object_parent_face_aabb)
        pad_selector_rows.append(tuple(pad_selectors))

        link_transform = transforms[pad.link_name]
        pad_envelope = pad_side_force_envelopes[finger_index]
        selected_center_local = tuple(
            model.addVar(
                lb=-1.0,
                ub=1.0,
                name=f"finger{finger_index}_pad_force_center_local{axis}",
            )
            for axis in range(3)
        )
        selected_cosine = model.addVar(
            lb=0.0,
            ub=1.0,
            name=f"finger{finger_index}_pad_force_cosine",
        )
        for axis in range(3):
            model.addCons(
                selected_center_local[axis]
                == quicksum(
                    pad_selectors[triangle] * float(
                        pad_envelope.center_local[triangle, axis]
                    )
                    for triangle in range(pad.triangle_count)
                ),
                name=f"finger{finger_index}_selected_pad_force_center{axis}",
            )
        model.addCons(
            selected_cosine
            == quicksum(
                pad_selectors[triangle]
                * float(pad_envelope.cosine_half_angle[triangle])
                for triangle in range(pad.triangle_count)
            ),
            name=f"finger{finger_index}_selected_pad_force_cosine",
        )
        center_hand = tuple(
            model.addVar(
                lb=-1.0,
                ub=1.0,
                name=f"finger{finger_index}_pad_force_center_hand{axis}",
            )
            for axis in range(3)
        )
        for axis in range(3):
            model.addCons(
                center_hand[axis]
                == sum(
                    link_transform[axis][column]
                    * selected_center_local[column]
                    for column in range(3)
                ),
                name=f"finger{finger_index}_pad_force_center_fk{axis}",
            )
        pad_force_centers_hand.append(center_hand)
        pad_force_cosines.append(selected_cosine)
        hand_point = tuple(
            model.addVar(lb=-hand_radius, ub=hand_radius, name=f"finger{finger_index}_ph{axis}")
            for axis in range(3)
        )
        for axis in range(3):
            model.addCons(
                hand_point[axis]
                == sum(link_transform[axis][column] * pad_point[column] for column in range(3))
                + link_transform[axis][3],
                name=f"finger{finger_index}_fk_contact_{axis}",
            )
            model.addCons(
                hand_point[axis]
                == sum(rotation_hc[axis][column] * object_point[column] for column in range(3))
                + translation_hc[axis],
                name=f"finger{finger_index}_shared_contact_{axis}",
            )
        hand_points.append(hand_point)

        jacobian = [[0.0 for _ in hand.independent_joint_names] for _ in range(3)]
        for joint_name in _ancestor_joint_names(hand, pad.link_name):
            joint = hand.joints[joint_name]
            if not joint.movable:
                continue
            joint_frame = _matmul(
                transforms[joint.parent_link], _numeric_matrix(joint.origin_transform())
            )
            axis_hand = [
                sum(joint_frame[row][column] * float(joint.axis[column]) for column in range(3))
                for row in range(3)
            ]
            lever = [hand_point[axis] - joint_frame[axis][3] for axis in range(3)]
            if joint.joint_type in ("revolute", "continuous"):
                linear = _cross(axis_hand, lever)
            elif joint.joint_type == "prismatic":
                linear = tuple(axis_hand)
            else:
                continue
            source, multiplier = _affine_source(hand, joint_name, affine_cache)
            column = joint_column[source]
            for axis in range(3):
                jacobian[axis][column] = jacobian[axis][column] + multiplier * linear[axis]
        linear_jacobians.append(jacobian)

    force_component_bound = contract.outer_single_contact_force_norm_cap_n
    gravity_force = np.asarray(contract.gravity_object_m_s2) * contract.mass_kg
    com = np.asarray(contract.center_of_mass_object_m, dtype=np.float64)

    for scenario_index, scenario in enumerate(scenarios):
        total_object_forces: list[tuple[Any, Any, Any]] = []
        total_hand_forces: list[tuple[Any, Any, Any]] = []
        for finger_index in range(contract.contact_count):
            total_force = tuple(
                model.addVar(
                    lb=-force_component_bound,
                    ub=force_component_bound,
                    name=f"s{scenario_index}_i{finger_index}_fc{axis}",
                )
                for axis in range(3)
            )
            model.addCons(
                quicksum(value * value for value in total_force)
                <= force_component_bound * force_component_bound,
                name=f"s{scenario_index}_i{finger_index}_force_ball",
            )
            for support_name, direction, support in (
                directional_force_bound.contact_force_axis_supports
            ):
                model.addCons(
                    quicksum(
                        float(direction[axis]) * total_force[axis]
                        for axis in range(3)
                    )
                    <= float(support),
                    name=(
                        f"s{scenario_index}_i{finger_index}_global_support_"
                        f"{support_name}"
                    ),
                )
            total_object_forces.append(total_force)

            hand_force = tuple(
                model.addVar(
                    lb=-force_component_bound,
                    ub=force_component_bound,
                    name=f"s{scenario_index}_i{finger_index}_fh{axis}",
                )
                for axis in range(3)
            )
            for axis in range(3):
                model.addCons(
                    hand_force[axis]
                    == -sum(rotation_hc[axis][column] * total_force[column] for column in range(3)),
                    name=f"s{scenario_index}_i{finger_index}_force_to_hand{axis}",
                )
            pad_force_norm = model.addVar(
                lb=0.0,
                ub=force_component_bound,
                name=f"s{scenario_index}_i{finger_index}_pad_force_norm",
            )
            model.addCons(
                quicksum(value * value for value in hand_force)
                <= pad_force_norm * pad_force_norm,
                name=f"s{scenario_index}_i{finger_index}_pad_force_norm_bound",
            )
            model.addCons(
                -quicksum(
                    pad_force_centers_hand[finger_index][axis]
                    * hand_force[axis]
                    for axis in range(3)
                )
                >= pad_force_cosines[finger_index] * pad_force_norm,
                name=f"s{scenario_index}_i{finger_index}_pad_unilateral_friction_outer",
            )
            total_hand_forces.append(hand_force)

        for axis in range(3):
            model.addCons(
                quicksum(force[axis] for force in total_object_forces)
                + float(gravity_force[axis])
                + gamma * scenario.force_object_n[axis]
                == 0.0,
                name=f"s{scenario_index}_force_equilibrium{axis}",
            )
        contact_moments = []
        for finger_index in range(contract.contact_count):
            lever = [object_points[finger_index][axis] - float(com[axis]) for axis in range(3)]
            contact_moments.append(_cross(lever, total_object_forces[finger_index]))
        for axis in range(3):
            model.addCons(
                quicksum(moment[axis] for moment in contact_moments)
                + gamma * scenario.torque_object_nm[axis]
                == 0.0,
                name=f"s{scenario_index}_moment_equilibrium{axis}",
            )

        for joint_index, joint_name in enumerate(hand.independent_joint_names):
            torque = model.addVar(
                lb=-contract.hand_drive_maximum_effort_nm,
                ub=contract.hand_drive_maximum_effort_nm,
                name=f"s{scenario_index}_tau_{joint_name}",
            )
            model.addCons(
                torque
                == quicksum(
                    linear_jacobians[finger_index][axis][joint_index]
                    * total_hand_forces[finger_index][axis]
                    for finger_index in range(contract.contact_count)
                    for axis in range(3)
                ),
                name=f"s{scenario_index}_joint_equilibrium_{joint_name}",
            )
            cap = contract.hand_drive_maximum_effort_nm
            model.addCons(torque <= cap, name=f"s{scenario_index}_{joint_name}_cap_pos")
            model.addCons(-torque <= cap, name=f"s{scenario_index}_{joint_name}_cap_neg")

    model.setObjective(-gamma, "minimize")
    return OuterMasterBundle(
        model=model,
        contract=contract,
        atlas=atlas,
        hand_contract=hand_contract,
        hand_model=hand,
        scenarios=scenarios,
        gamma=gamma,
        quaternion=quaternion,
        translation_hc=translation_hc,
        q_pre=q_pre,
        q_contact=q_contact,
        q_goal=q_goal,
        closing_order_selectors=closing_order_selectors,
        object_contact_points=tuple(tuple(row) for row in object_points),
        pad_contact_points=tuple(tuple(row) for row in pad_points),
        object_parent_face_aabb_variables=(
            object_parent_face_aabb_rows[0],
            object_parent_face_aabb_rows[1],
            object_parent_face_aabb_rows[2],
        ),
        pad_triangle_selectors=(
            pad_selector_rows[0],
            pad_selector_rows[1],
            pad_selector_rows[2],
        ),
        object_aabb_lower_m=tuple(float(value) for value in object_aabb_lower),
        object_aabb_upper_m=tuple(float(value) for value in object_aabb_upper),
        translation_bound_m=translation_bound,
        force_ball_gamma_upper_bound=force_ball_gamma_upper_bound,
        directional_force_bound=directional_force_bound,
        pad_side_force_envelopes=(
            pad_side_force_envelopes[0],
            pad_side_force_envelopes[1],
            pad_side_force_envelopes[2],
        ),
    )


def _finite_or_none(value: Any) -> float | None:
    parsed = float(value)
    return parsed if math.isfinite(parsed) and abs(parsed) < 1.0e19 else None


def _outer_incumbent_report(bundle: OuterMasterBundle) -> dict[str, Any]:
    """Expose a relaxation witness without promoting it to a grasp candidate."""

    model = bundle.model
    searchable = frozenset(bundle.atlas.contract.proven_searchable_parent_faces)
    finger_rows: list[dict[str, Any]] = []
    for finger_index, (object_row, pad_selectors) in enumerate(
        zip(
            bundle.object_parent_face_aabb_variables,
            bundle.pad_triangle_selectors,
            strict=True,
        )
    ):
        parent_values = [float(model.getVal(value)) for value in object_row.selectors]
        parent_row = int(np.argmax(parent_values))
        parent_face = int(object_row.parent_face_indices[parent_row])
        pad_values = [float(model.getVal(value)) for value in pad_selectors]
        pad_triangle = int(np.argmax(pad_values))
        finger_rows.append(
            {
                "finger_index_zero_based": finger_index,
                "object_point_object_frame_m": [
                    float(model.getVal(value))
                    for value in bundle.object_contact_points[finger_index]
                ],
                "selected_parent_face_aabb_label_one_based": parent_face,
                "selected_parent_face_semantic": (
                    "PROVEN_SEARCHABLE"
                    if parent_face in searchable
                    else "UNRESOLVED_RETAINED_IN_OUTER_DOMAIN"
                ),
                "selected_parent_face_aabb_lower_m": list(
                    object_row.lower_m[parent_row]
                ),
                "selected_parent_face_aabb_upper_m": list(
                    object_row.upper_m[parent_row]
                ),
                "selected_parent_selector_value": parent_values[parent_row],
                "parent_selector_sum": float(sum(parent_values)),
                "pad_point_link_local_m": [
                    float(model.getVal(value))
                    for value in bundle.pad_contact_points[finger_index]
                ],
                "selected_pad_contract_triangle_index_zero_based": pad_triangle,
                "selected_pad_selector_value": pad_values[pad_triangle],
                "pad_selector_sum": float(sum(pad_values)),
                "parent_face_label_is_not_a_surface_contact_claim": True,
            }
        )
    closing_order_values = {
        "_then_".join(order): float(model.getVal(selector))
        for order, selector in bundle.closing_order_selectors.items()
    }
    return {
        "role": (
            "OUTER_RELAXATION_PRIMAL_FOR_DIAGNOSING_THE_NEXT_RELAXATION_GAP_"
            "NOT_A_CONTACT_OR_EXECUTABLE_GRASP"
        ),
        "gamma_ae": float(model.getVal(bundle.gamma)),
        "q_pre_rad": {
            name: float(model.getVal(bundle.q_pre[name]))
            for name in bundle.hand_model.independent_joint_names
        },
        "q_contact_outer_proxy_rad": {
            name: float(model.getVal(bundle.q_contact[name]))
            for name in bundle.hand_model.independent_joint_names
        },
        "q_goal_contact_search_endpoint_rad": {
            name: float(model.getVal(bundle.q_goal[name]))
            for name in bundle.hand_model.independent_joint_names
        },
        "quaternion_hc_wxyz": [
            float(model.getVal(value)) for value in bundle.quaternion
        ],
        "translation_hc_m": [
            float(model.getVal(value)) for value in bundle.translation_hc
        ],
        "closing_order_symmetry_selector_values_not_executable_order": (
            closing_order_values
        ),
        "fingers": finger_rows,
    }


def _axial_endpoint_incumbent_report(
    bundle: AxialEndpointOuterBundle,
) -> dict[str, Any]:
    model = bundle.model
    q_pre = [
        float(model.getVal(bundle.q_pre[name]))
        for name in bundle.hand_model.independent_joint_names
    ]
    q_goal = [
        float(model.getVal(bundle.q_goal[name]))
        for name in bundle.hand_model.independent_joint_names
    ]
    quaternion = [float(model.getVal(value)) for value in bundle.quaternion]
    translation = [float(model.getVal(value)) for value in bundle.translation_hc]
    scenario_rows: list[dict[str, Any]] = []
    for scenario_index, axial_error in enumerate(bundle.axial_errors_m):
        q_contact = [
            float(
                model.getVal(
                    bundle.q_contact_by_scenario[scenario_index][name]
                )
            )
            for name in bundle.hand_model.independent_joint_names
        ]
        contacts: list[dict[str, Any]] = []
        for finger_index in range(3):
            object_surface = bundle.object_surfaces[scenario_index][finger_index]
            pad_surface = bundle.pad_surfaces[scenario_index][finger_index]
            object_values = np.asarray(
                [float(model.getVal(value)) for value in object_surface.selectors]
            )
            pad_values = np.asarray(
                [float(model.getVal(value)) for value in pad_surface.selectors]
            )
            object_index = int(np.argmax(object_values))
            pad_index = int(np.argmax(pad_values))
            object_barycentric = [
                float(model.getVal(value))
                for value in object_surface.barycentric[object_index]
            ]
            pad_barycentric = [
                float(model.getVal(value))
                for value in pad_surface.barycentric[pad_index]
            ]
            contacts.append(
                {
                    "finger": finger_index + 1,
                    "object_triangle_zero_based": object_index,
                    "object_parent_face_one_based": int(
                        bundle.atlas.parent_face_index[object_index]
                    ),
                    "object_barycentric": object_barycentric,
                    "pad_triangle_zero_based": pad_index,
                    "pad_barycentric": pad_barycentric,
                }
            )
        scenario_rows.append(
            {
                "axial_pose_error_m": axial_error,
                "q_contact_rad": q_contact,
                "contacts": contacts,
            }
        )
    return {
        "q_pre_rad": q_pre,
        "q_goal_rad": q_goal,
        "quaternion_hc_wxyz": quaternion,
        "translation_hc_m": translation,
        "endpoint_responses": scenario_rows,
    }


def solve_axial_endpoint_outer_master(
    bundle: AxialEndpointOuterBundle,
    *,
    time_limit_s: float,
    node_limit: int | None,
) -> dict[str, Any]:
    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0:
        raise AnalyticOuterMasterError("time limit must be finite and positive")
    if node_limit is not None and node_limit < 1:
        raise AnalyticOuterMasterError("node limit must be positive")
    model = bundle.model
    model.setParam("display/verblevel", 0)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("randomization/permutationseed", 0)
    model.setParam("limits/time", float(time_limit_s))
    if node_limit is not None:
        model.setParam("limits/nodes", int(node_limit))
    model.optimize()
    solution_count = int(model.getNSols())
    report: dict[str, Any] = {
        "schema_version": "kcg_te_axial_endpoint_zero_motion_seed_screen_v2",
        "claim_scope": "SIMULATION_ONLY_FLOATING_POINT_ZERO_RELATIVE_MOTION_MODE_SEED_SCREEN",
        "hardware_authorized": False,
        "development_object": EXPECTED_DEVELOPMENT_OBJECT,
        "status": str(model.getStatus()),
        "model_role": "ZERO_RELATIVE_MOTION_TWO_ENDPOINT_INTERSECTION_SCREEN",
        "pruning_authorized": False,
        "contact_transform_assumption": (
            "T_contact_at_endpoint=T_HC_pre*Trans_C(0,0,axial_error); "
            "passive object and palm-base motion during closure omitted"
        ),
        "contact_joint_assumption": (
            "actual palm q_contact equals q_pre and actual closing q_contact "
            "lies componentwise between position targets q_pre and q_goal"
        ),
        "axial_pose_errors_m": list(bundle.axial_errors_m),
        "shared_design_variables": ["T_HC_pre", "q_pre", "q_goal"],
        "scenario_specific_response_variables": [
            "q_contact",
            "object_parent_and_triangle",
            "object_uv_barycentric",
            "pad_triangle_and_barycentric",
        ],
        "object_surface_semantics": "EXACT_PLANE_OR_CYLINDER_PARENT_OVER_CLOSED_UV_TRIANGLE",
        "object_domain_boundary": (
            "FROZEN_UV_CHART_MAPPED_TO_ANALYTIC_CARRIERS_NOT_PROVED_TO_CONTAIN_"
            "THE_FROZEN_MESH_OR_COMPLETE_TRIMMED_BREP_FACES"
        ),
        "pad_surface_semantics": "FROZEN_COMPLETE_PAD_CLOSED_TRIANGLE_UNION",
        "object_triangle_modes_per_finger_per_endpoint": bundle.atlas.triangle_count,
        "pad_triangle_modes_per_finger_per_endpoint": [
            pad.triangle_count for pad in bundle.hand_contract.pads
        ],
        "number_of_variables": int(model.getNVars()),
        "number_of_constraints": int(model.getNConss()),
        "solution_count": solution_count,
        "time_limit_s": float(time_limit_s),
        "node_limit": node_limit,
        "translation_hc_bound_m": bundle.translation_bound_m,
        "intersection_witness_not_legal_grasp": None,
        "screen_assumptions_and_omissions": [
            "zero_passive_hand_object_relative_motion_during_contact_establishment",
            "arm_base_compliance_motion_during_contact_establishment",
            "finite_stiffness_contact_deflection_outside_the_target_interval",
            "continuous_axial_interval_between_endpoints",
            "other_five_pose_error_coordinates",
            "parent_and_pad_normal_opposition",
            "first_contact_and_monotone_closure_response",
            "forbidden_surface_clearance_61um",
            "global_nonpenetration_self_and_table_collision",
            "force_friction_pd_and_effort_feasibility",
            "arm_ik_and_paths",
            "Isaac_dynamics_lift_and_hold",
        ],
        "not_claimed": [
            "necessary_condition_for_full_continuous_grasp_domain_X",
            "safe_global_contact_mode_or_parent_surface_pruning",
            "necessary_outer_for_the_controller_reachable_contact_domain",
            "infeasibility_pruning_of_the_frozen_mesh_or_trimmed_brep_domain",
            "continuous_U_robust_feasible_candidate",
            "finite_UB_AE",
            "global_real_arithmetic_infeasibility",
            "legal_contact_or_grasp",
            "50mm_lift_or_2s_hold",
            "hardware_validity",
        ],
    }
    if solution_count:
        report["intersection_witness_not_legal_grasp"] = (
            _axial_endpoint_incumbent_report(bundle)
        )
    return report


def solve_outer_master(
    bundle: OuterMasterBundle,
    *,
    time_limit_s: float,
    node_limit: int | None,
) -> dict[str, Any]:
    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0:
        raise AnalyticOuterMasterError("time limit must be finite and positive")
    if node_limit is not None and node_limit < 1:
        raise AnalyticOuterMasterError("node limit must be positive")
    model = bundle.model
    model.setParam("display/verblevel", 0)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("randomization/permutationseed", 0)
    model.setParam("limits/time", float(time_limit_s))
    if node_limit is not None:
        model.setParam("limits/nodes", int(node_limit))
    model.optimize()

    status = str(model.getStatus())
    dual_bound = _finite_or_none(model.getDualbound())
    primal_bound = _finite_or_none(model.getPrimalbound())
    solution_count = int(model.getNSols())
    outer_gamma = None
    if solution_count > 0:
        outer_gamma = float(model.getVal(bundle.gamma))
    force_cap_lb = -bundle.contract.force_only_gamma_upper_bound
    directional_lb = -bundle.directional_force_bound.gamma_upper_bound
    available_lower_bounds = {
        "force_only_capacity": force_cap_lb,
        "global_axis_support": directional_lb,
    }
    if dual_bound is not None:
        available_lower_bounds["floating_point_scip_dual"] = dual_bound
    strongest_lower_bound_source, strongest_lower_bound = max(
        available_lower_bounds.items(), key=lambda row: row[1]
    )
    parent_face_indices = (
        bundle.object_parent_face_aabb_variables[0].parent_face_indices
    )
    if any(
        row.parent_face_indices != parent_face_indices
        for row in bundle.object_parent_face_aabb_variables[1:]
    ):
        raise AnalyticOuterMasterError("per-finger parent-face AABB modes differ")
    searchable_parent_faces = frozenset(
        bundle.atlas.contract.proven_searchable_parent_faces
    )
    searchable_triangle_mask = np.isin(
        bundle.atlas.parent_face_index,
        tuple(searchable_parent_faces),
    )
    report: dict[str, Any] = {
        "schema_version": "kcg_te_analytic_outer_master_compact_root_result_v5",
        "claim_scope": CLAIM_SCOPE,
        "hardware_authorized": False,
        "development_object": EXPECTED_DEVELOPMENT_OBJECT,
        "status": status,
        "objective": "J_AE=-gamma_AE",
        "model_role": (
            "FINITE_SCENARIO_PARENT_FACE_AABB_UNION_AXIS_SUPPORT_AND_PAD_"
            "UNILATERAL_SINGLE_POINT_OUTER_RELAXATION"
        ),
        "dual_bound_lb_ae": dual_bound,
        "capacity_upper_from_dual": None if dual_bound is None else -dual_bound,
        "strongest_available_lb_ae": strongest_lower_bound,
        "strongest_available_lb_ae_source": strongest_lower_bound_source,
        "available_lb_ae": available_lower_bounds,
        "outer_model_primal_bound_not_ub_ae": primal_bound,
        "outer_model_incumbent_gamma_not_candidate": outer_gamma,
        "ub_ae": None,
        "gap_ae": None,
        "force_only_gamma_upper_bound": bundle.contract.force_only_gamma_upper_bound,
        "force_only_j_ae_lower_bound": force_cap_lb,
        "global_force_ball_gamma_upper_bound": (
            bundle.force_ball_gamma_upper_bound
        ),
        "global_force_ball_single_contact_radius_n": (
            bundle.contract.outer_single_contact_force_norm_cap_n
        ),
        "global_axis_support_gamma_upper_bound": (
            bundle.directional_force_bound.gamma_upper_bound
        ),
        "global_axis_support_j_ae_lower_bound": directional_lb,
        "global_axis_support_limiting_scenario": (
            bundle.directional_force_bound.limiting_scenario
        ),
        "global_axis_contact_force_supports": [
            {
                "source_disturbance_scenario": name,
                "contact_force_direction": list(direction),
                "single_contact_support_n": support,
            }
            for name, direction, support in (
                bundle.directional_force_bound.contact_force_axis_supports
            )
        ],
        "dual_improves_force_only_bound": (
            None
            if dual_bound is None
            else dual_bound > force_cap_lb + 1.0e-9
        ),
        "dual_improves_global_axis_support_bound": (
            None
            if dual_bound is None
            else dual_bound > directional_lb + 1.0e-9
        ),
        "scenario_count": len(bundle.scenarios),
        "scenario_names": [row.name for row in bundle.scenarios],
        "object_triangle_count_covered_by_parent_face_aabb_union_per_finger": (
            bundle.atlas.triangle_count
        ),
        "object_parent_face_aabb_modes_per_finger": len(parent_face_indices),
        "object_mode_variables_per_finger": len(parent_face_indices),
        "object_global_containing_aabb_lower_m": list(bundle.object_aabb_lower_m),
        "object_global_containing_aabb_upper_m": list(bundle.object_aabb_upper_m),
        "object_global_aabb_is_coordinate_envelope_not_set_equality": True,
        "object_parent_face_aabb_union_contains_every_frozen_nonhard_triangle": (
            True
        ),
        "parent_face_aabbs_outward_rounded_one_float_ulp": True,
        "object_parent_faces_equal_all_frozen_nonhard_parent_faces": (
            parent_face_indices == tuple(bundle.atlas.contract.allowed_parent_faces)
        ),
        "object_proven_searchable_parent_face_count": len(
            searchable_parent_faces
        ),
        "object_unresolved_parent_face_count": (
            len(parent_face_indices) - len(searchable_parent_faces)
        ),
        "object_proven_searchable_triangle_count": int(
            np.count_nonzero(searchable_triangle_mask)
        ),
        "object_unresolved_triangle_count": int(
            np.count_nonzero(~searchable_triangle_mask)
        ),
        "parent_faces_500_and_502_retained": (
            500 in parent_face_indices and 502 in parent_face_indices
        ),
        "unresolved_faces_retained_without_promotion_to_proven_searchable": True,
        "parent_face_exact_one_is_computational_partition_not_permission": True,
        "overlapping_parent_aabbs_and_shared_boundaries_retained": True,
        "shared_parent_boundaries_intrinsically_banned": False,
        "parent_face_aabb_may_contain_trim_void_or_hard_geometry": True,
        "aabb_and_object_point_coordinate_frame": "FROZEN_CONNECTOR_OBJECT_FRAME",
        "exact_trimmed_step_brep_containment_claimed": False,
        "pad_triangle_modes_per_finger": [
            pad.triangle_count for pad in bundle.hand_contract.pads
        ],
        "pad_continuous_barycentric_coordinates": True,
        "pad_side_unilateral_friction_outer_envelope_added": True,
        "pad_edge_vertex_normal_cones_outer_enveloped_by_triangle_vertex_stars": True,
        "pad_vertex_stars_too_broad_for_convex_directional_cone_fully_relaxed": True,
        "pad_side_force_envelopes": [
            {
                "pad_name": pad.name,
                "triangle_count": pad.triangle_count,
                "directionally_constrained_triangle_count": (
                    envelope.constrained_triangle_count
                ),
                "directionally_unconstrained_triangle_count": (
                    envelope.unconstrained_triangle_count
                ),
                "circumscribed_friction_half_angle_rad": (
                    envelope.friction_half_angle_rad
                ),
            }
            for pad, envelope in zip(
                bundle.hand_contract.pads,
                bundle.pad_side_force_envelopes,
                strict=True,
            )
        ],
        "object_point_surface_membership_relaxed_to_parent_face_aabb_union": True,
        "object_contact_representation": "ONE_POINT_PROXY_PER_FINGER",
        "distributed_multiface_contact_physics_outer_covered": False,
        "object_location_normal_coupling_omitted": True,
        "object_normal_force_coupling_relaxed_to_global_force_ball": True,
        "global_axis_force_support_halfspaces_added": True,
        "complete_nonpad_terminal_collision_omitted_from_lower_bound": True,
        "closing_order_decision": {
            "mode": "EXACTLY_ONE_OF_ALL_SIX_FINGER_PERMUTATIONS",
            "orders": [list(order) for order in bundle.contract.closing_orders],
            "fixed_finger_1_first": False,
            "selector_count": len(bundle.closing_order_selectors),
            "outer_selector_values_are_symmetry_labels_only": True,
            "outer_selectors_are_not_executable_order_or_inner_witness": True,
        },
        "active_finger_first_contact_rule": (
            bundle.contract.active_finger_first_contact_rule
        ),
        "closure_order_contact_timing_omitted_from_lower_bound": True,
        "pad_before_nonpad_must_be_proved_by_inner_feasibility": True,
        "visual_pose_empirical_hard_set_defined": (
            bundle.contract.visual_pose_empirical_hard_set_defined
        ),
        "visual_pose_empirical_hard_set_status": (
            bundle.contract.visual_pose_empirical_hard_set_status
        ),
        "visual_pose_research_set_defined": (
            bundle.contract.visual_pose_research_set_defined
        ),
        "visual_pose_research_set_frozen_before_robust_grasp_outcomes": (
            bundle.contract.visual_pose_research_set_frozen_before_robust_grasp_outcomes
        ),
        "visual_pose_research_set_role": (
            bundle.contract.visual_pose_research_set_role
        ),
        "visual_pose_research_set_source": (
            bundle.contract.visual_pose_research_set_source
        ),
        "visual_pose_research_set_lower": list(
            bundle.contract.visual_pose_research_set_lower
        ),
        "visual_pose_research_set_upper": list(
            bundle.contract.visual_pose_research_set_upper
        ),
        "visual_pose_research_set_guarantee_scope": (
            bundle.contract.visual_pose_research_set_guarantee_scope
        ),
        "visual_pose_research_set_coverage_probability": None,
        "visual_pose_research_set_transfer_limit": (
            bundle.contract.visual_pose_research_set_transfer_limit
        ),
        "visual_pose_measurement_protocol_executable": (
            bundle.contract.visual_pose_measurement_protocol_executable
        ),
        "visual_pose_measurement_protocol_status": (
            bundle.contract.visual_pose_measurement_protocol_status
        ),
        "visual_pose_residual_transform": (
            bundle.contract.visual_pose_residual_transform
        ),
        "visual_pose_failure_rule": bundle.contract.visual_pose_failure_rule,
        "visual_pose_empirical_support_rule": (
            bundle.contract.visual_pose_empirical_support_rule
        ),
        "visual_pose_miss_rule": bundle.contract.visual_pose_miss_rule,
        "visual_pose_empirical_support_claim": (
            bundle.contract.visual_pose_empirical_support_claim
        ),
        "old_pose_radii_role": bundle.contract.stress_test_pose_role,
        "stress_test_pose_translation_radius_m": (
            bundle.contract.stress_test_pose_translation_radius_m
        ),
        "stress_test_pose_rotation_radius_rad": (
            bundle.contract.stress_test_pose_rotation_radius_rad
        ),
        "shared_design_across_scenarios": True,
        "scenario_specific_contact_force_only": True,
        "analytic_joint_torque_capacity": {
            "native_hand_drive_cap_nm": (
                bundle.contract.hand_drive_maximum_effort_nm
            ),
            "applied_to_active_hand_joints": list(
                bundle.contract.independent_joint_names
            ),
            "role": "OUTER_ACTUATOR_CAPACITY_ONLY_NOT_REALIZED_PD_RESPONSE",
        },
        "runtime_effort_thresholds_not_static_torque_caps": {
            "contact_effort_rise_nm": bundle.contract.contact_effort_rise_nm,
            "contact_effort_rise_phase_scope": (
                "SEQUENTIAL_CONTACT_CONFIRMATION_AND_PRELIFT_CONTACT_GATE"
            ),
            "measured_effort_abort_nm": bundle.contract.measured_effort_abort_nm,
            "measured_effort_abort_phase_scope": (
                "PRELOAD_PRELIFT_CHECK_LIFT_AND_HOLD_ONLY"
            ),
            "measured_effort_abort_joint_scope": (
                "THREE_CLOSING_JOINTS_NOT_PALM_JOINT"
            ),
            "required_closing_joint_effort_role": (
                bundle.contract.required_closing_joint_effort_role
            ),
            "preload_closed_loop_response_omitted": True,
        },
        "position_target_semantics": {
            "q_goal_role": "CONTACT_SEARCH_ENDPOINT_NOT_HOLD_TARGET",
            "q_cmd_hold_is_closed_loop_response": True,
            "q_cmd_hold_closed_loop_response_omitted": True,
        },
        "translation_hc_bound_m": bundle.translation_bound_m,
        "number_of_variables": int(model.getNVars()),
        "number_of_constraints": int(model.getNConss()),
        "solution_count": solution_count,
        "time_limit_s": float(time_limit_s),
        "node_limit": node_limit,
        "contract": {
            "path": str(bundle.contract.path.relative_to(bundle.contract.repository_root)),
            "sha256": _sha256_file(bundle.contract.path),
        },
        "step_contact_atlas_sha256": bundle.atlas.atlas_sha256,
        "omitted_for_outer_relaxation": list(
            bundle.contract.omitted_for_outer_relaxation
        ),
        "not_claimed": [
            "continuous_U_robust_feasible_candidate",
            "actual_PD_capacity",
            "required_effort_threshold_is_a_joint_torque_cap",
            "runtime_projected_effort_gate_is_J_transpose_f",
            "dynamic_stability_or_antislip",
            "outer_model_primal_is_an_object_surface_contact",
            "outer_bound_for_distributed_multiface_physx_contact_physics",
            "legal_collision_free_grasp",
            "50mm_lift_or_2s_hold",
            "exact_STEP_real_arithmetic_certificate",
            "hardware_validity",
        ],
    }
    if solution_count:
        report["outer_incumbent_not_candidate"] = _outer_incumbent_report(bundle)
    return report


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "src/kcg_connector/config/te_continuous_grasp_analytic_envelope_v1.yaml"
        ),
    )
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--node-limit", type=int)
    parser.add_argument(
        "--axial-endpoint-outer",
        action="store_true",
        help=(
            "solve the zero-relative-motion +/- axial-pose mode-seed screen "
            "instead of the wrench outer"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    root = arguments.repository_root.resolve(strict=True)
    contract = load_analytic_envelope_contract(
        arguments.contract, repository_root=root
    )
    if arguments.axial_endpoint_outer:
        report = solve_axial_endpoint_outer_master(
            build_axial_endpoint_outer_master(contract),
            time_limit_s=arguments.time_limit_s,
            node_limit=arguments.node_limit,
        )
    else:
        report = solve_outer_master(
            build_outer_master(contract),
            time_limit_s=arguments.time_limit_s,
            node_limit=arguments.node_limit,
        )
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = root / output
        _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalyticEnvelopeContract",
    "AnalyticOuterMasterError",
    "AxialEndpointOuterBundle",
    "DirectionalForceGammaBound",
    "OuterMasterBundle",
    "WrenchScenario",
    "build_axial_endpoint_outer_master",
    "build_outer_master",
    "frozen_outer_scenarios",
    "load_analytic_envelope_contract",
    "solve_axial_endpoint_outer_master",
    "solve_outer_master",
]
