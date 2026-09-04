#!/usr/bin/env python3

"""Run a bounded, truth-isolated CARTS-Grasp V2 preflight or grasp-lift."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any, Mapping, Sequence
import uuid
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from . import controller as control
    from .engine_health import (
        PhysxStatsMonitor, current_engine_log_path, finalize_engine_evaluation,
        gpu_backend_record, gpu_world_parameters, identity_hashes_match,
        load_runtime_resources, preflight_is_accepted, synchronize_engine_log,
    )
    from .evaluate_run import (
        IsolatedHandRecorder, TruthAuditRecorder, audit_initial_joint_state,
        audit_mimic_schema, compare_reference_targets,
        evaluate_isolated_hand_trace, evaluate_trace,
    )
else:
    import controller as control
    from engine_health import (
        PhysxStatsMonitor, current_engine_log_path, finalize_engine_evaluation,
        gpu_backend_record, gpu_world_parameters, identity_hashes_match,
        load_runtime_resources, preflight_is_accepted, synchronize_engine_log,
    )
    from evaluate_run import (
        IsolatedHandRecorder, TruthAuditRecorder, audit_initial_joint_state,
        audit_mimic_schema, compare_reference_targets,
        evaluate_isolated_hand_trace, evaluate_trace,
    )
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    generate_axial_pad_intersection_grasp,
)
from kcg_connector.grasp.robust.object_model import file_sha256
from kcg_connector.robot_model import MIMIC_HAND_JOINTS
from kcg_connector.te_transport_grasp_target import (
    build_visual_transport_target,
    load_transport_grasp_relation,
    load_visual_transport_target,
)
from kcg_connector.virtual_wrist_ft_runtime import transform_wrench_to_task


ROBOT_ROOT = "/World/HandArm"
ARTICULATION_PATH = ROBOT_ROOT + "/Geometry/world"
HAND_BASE_PATH = ARTICULATION_PATH + (
    "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/"
    "iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link"
)
EXPECTED_DOF_NAMES = control.ARM_JOINT_NAMES + (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
TENSOR_CONTACT_MAX_COUNT = 4096
FINGERTIP_PHYSICS_MATERIAL_PATH = ROBOT_ROOT + "/PhysicsMaterials/fingertip_pad"
CONNECTOR_CONTACT_MATERIAL_PATH = "/World/CartsV2ConnectorContactMaterial"
CLOSING_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")
POSTGRASP_DISTURBANCE_SCHEMA = "te_postgrasp_disturbance_panel_v1"
POSTGRASP_DISTURBANCE_COM_SCHEMA = "te_postgrasp_disturbance_panel_v2"
POSTGRASP_DISTURBANCE_PHASES = (
    "postgrasp_disturbance_baseline",
    "postgrasp_disturbance_ramp_up",
    "postgrasp_disturbance_plateau",
    "postgrasp_disturbance_ramp_down",
    "postgrasp_disturbance_recovery",
)
HAND_TILT_PERTURBATION_KEY = "hand_tilt_about_object_pivot"
HAND_TILT_REQUIRED_KEYS = frozenset((
    "axis_supplier_object",
    "angle_rad",
    "pivot_supplier_object_m",
))
SAME_RESET_RGBD_MODE = "same-reset-rgbd-observe"
SAME_RESET_RGBD_SCHEMA = "kcg_te_same_reset_rgbd_observe_v1"
VISION_HIGH_REOBSERVE_MODE = "vision-high-reobserve"
VISION_GRASP_SERVO_MODE = "vision-grasp-servo"
VISION_MOTION_MODES = (
    VISION_HIGH_REOBSERVE_MODE,
    VISION_GRASP_SERVO_MODE,
)
VISION_HIGH_REOBSERVE_SCHEMA = "kcg_te_visual_high_reobserve_v1"
VISION_HIGH_REOBSERVE_SLOW_SCHEMA = "kcg_te_visual_high_reobserve_slow_v1"
VISION_HIGH_REOBSERVE_SLOW2_SCHEMA = "kcg_te_visual_high_reobserve_slow2_v1"
VISION_HIGH_REOBSERVE_DYNAMIC_INERTIA_SCHEMA = (
    "kcg_te_visual_high_reobserve_dynamic_inertia_v1"
)
VISION_SECOND_SHORT_PREFIX_SCHEMA = "kcg_te_visual_second_short_prefix_v1"
VISION_SECOND_SHORT_PREFIX_EXACT_SCHEMA = (
    "kcg_te_visual_second_short_prefix_exact_v1"
)
VISION_SECOND_SHORT_PREFIX_EXACT_OBSTACLES_SCHEMA = (
    "kcg_te_visual_second_short_prefix_exact_obstacles_v1"
)
FREE_SPLIT_SCENE_KIND = "FREE_SPLIT_REVOLUTE_ON_SHARED_FINITE_TABLE"


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_repository_file(
    repository: Path,
    record: Mapping[str, object],
    label: str,
) -> Path:
    path = (repository / str(record["path"])).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} identity differs")
    return path


def _load_visual_high_reobserve_contract(
    repository: Path,
    path: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_same_reset_config",
        "frozen_grasp_relation",
        "source_evidence",
        "simulation_research_range",
        "static_path_panel",
        "motion",
        "wrist_ft_safety",
        "reobservation",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_HIGH_REOBSERVE_SCHEMA
    ):
        raise ValueError("visual high-reobserve config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_FRESH_VISION_TO_APPROACH_HIGH_AND_REOBSERVE"
    ):
        raise ValueError("visual high-reobserve mode contract differs")
    if document["authorization"] != {
        "hardware_authorized": False,
        "simulation_only": True,
        "approach_high_motion_authorized": True,
        "descent_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("visual high-reobserve authorization differs")

    base_path = _bound_repository_file(
        repository, document["base_same_reset_config"], "base same-reset config"
    )
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    relation_path = _bound_repository_file(
        repository, document["frozen_grasp_relation"], "transport grasp relation"
    )
    load_transport_grasp_relation(relation_path, repository)
    evidence_paths = {
        name: _bound_repository_file(repository, record, f"high-mode {name}")
        for name, record in document["source_evidence"].items()
    }
    if set(evidence_paths) != {
        "same_reset_provider",
        "same_reset_sealed_capture",
        "same_reset_posthoc",
        "translated_ring_provider",
        "translated_ring_posthoc",
        "hand_inertial_source",
    }:
        raise ValueError("visual high-reobserve source evidence set differs")
    sealed = json.loads(
        evidence_paths["same_reset_sealed_capture"].read_text(encoding="utf-8")
    )
    same_reset_posthoc = json.loads(
        evidence_paths["same_reset_posthoc"].read_text(encoding="utf-8")
    )
    translated_posthoc = json.loads(
        evidence_paths["translated_ring_posthoc"].read_text(encoding="utf-8")
    )
    if (
        sealed.get("image_and_freshness_gate_pass") is not True
        or sealed.get("provider_result_sealed_before_truth_read") is not True
        or same_reset_posthoc.get("posthoc_research_gate_pass") is not True
        or same_reset_posthoc.get("provider_result_sealed_before_truth_read")
        is not True
        or translated_posthoc.get(
            "independent_translated_lateral_validation_pass"
        )
        is not True
        or translated_posthoc.get(
            "lateral_simulation_research_bound_freeze_supported"
        )
        is not True
        or translated_posthoc.get("control_authorized") is not False
    ):
        raise ValueError("visual high-reobserve bound research evidence differs")

    research = document["simulation_research_range"]
    if research != {
        "kind": "FINITE_ADMISSIBLE_SET_NOT_3SIGMA",
        "lateral_x_absolute_m": 0.000183,
        "lateral_y_absolute_m": 0.000183,
        "support_z_absolute_m": 0.003,
        "axis_tilt_cone_rad": 0.08726646259971647,
        "yaw_range_rad": [-math.pi, math.pi],
        "yaw_consumed_by_transport_target": False,
    }:
        raise ValueError("visual high-reobserve finite research range differs")
    panel = document["static_path_panel"]
    expected_panel = {
        "method": (
            "COMPLETE_17_LINK_FCL_126_STATE_FINITE_PANEL_PLUS_CONTINUOUS_"
            "OBJECT_POSE_ENVELOPE"
        ),
        "collision_link_count": 17,
        "state_count": 126,
        "approach_high_clearance_m": 0.05,
        "minimum_table_clearance_m": 0.024490201473236106,
        "minimum_nominal_plug_clearance_m": 0.029987321248709693,
        "minimum_fixture_clearance_m": 0.2948251196787564,
        "maximum_object_pose_surface_displacement_m": 0.006973043655673907,
        "minimum_range_robust_plug_clearance_m": 0.023014277593035787,
        "mathematical_continuous_joint_space_proof_claimed": False,
    }
    if panel != expected_panel or not math.isclose(
        float(panel["minimum_nominal_plug_clearance_m"])
        - float(panel["maximum_object_pose_surface_displacement_m"]),
        float(panel["minimum_range_robust_plug_clearance_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("visual high-reobserve static path panel differs")
    motion = document["motion"]
    if motion != {
        "allowed_controller_phases": [
            "ft_free_space_tare",
            "settle",
            "preshape_at_home",
            "approach_above",
            "wait_above_settled",
            "observation_wait_hold",
        ],
        "observation_wait_hold_duration_s": 0.5,
        "required_descent_command_count": 0,
        "required_finger_closure_command_count": 0,
    }:
        raise ValueError("visual high-reobserve motion contract differs")
    ft = document["wrist_ft_safety"]
    if ft != {
        "source_joint": "hand2arm",
        "reaction_row_rule": "METADATA_JOINT_INDEX_PLUS_ONE",
        "canonical_from_raw": "NEGATIVE_IDENTITY_6",
        "tare_duration_s": 0.5,
        "minimum_tare_samples": 100,
        "gravity_difference_compensation": (
            "FROZEN_HAND_LINK_MASS_COM_MODEL_FROM_HAND_XACRO"
        ),
        "maximum_resultant_force_n": 3.0400615,
        "maximum_resultant_torque_nm": 0.0540690202,
        "limit_source": (
            "USER_GOAL_SIMULATION_ONLY_RESEARCH_UPPER_BOUND_NOT_HARDWARE_"
            "CALIBRATION"
        ),
        "hardware_load_certification_claimed": False,
        "threshold_action": "STOP_AND_HOLD_NO_DESCENT",
    }:
        raise ValueError("visual high-reobserve wrist FT contract differs")
    if document["reobservation"] != {
        "same_reset_required": True,
        "robot_stationary_required": True,
        "provider_scope": "TRANSPORT_PLUG_ONLY",
        "miss_result": "REOBSERVATION_MISS_STOP_NO_DESCENT",
        "truth_read_only_after_second_provider_sealed": True,
    }:
        raise ValueError("visual high-reobserve second observation contract differs")

    return base_path, base_document


def _load_visual_high_reobserve_slow_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_visual_high_reobserve_config",
        "direct_failure_evidence",
        "single_variable_override",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_HIGH_REOBSERVE_SLOW_SCHEMA
    ):
        raise ValueError("visual high-reobserve slow config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_FRESH_VISION_TO_APPROACH_HIGH_AND_REOBSERVE"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "approach_high_motion_authorized": True,
        "descent_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("visual high-reobserve slow authorization differs")
    base_path = _bound_repository_file(
        repository,
        document["base_visual_high_reobserve_config"],
        "base visual high-reobserve config",
    )
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    evidence = document["direct_failure_evidence"]
    if set(evidence) != {"high_motion_execution", "trace"}:
        raise ValueError("visual high-reobserve slow evidence set differs")
    high_motion_path = _bound_repository_file(
        repository, evidence["high_motion_execution"], "run01 high motion"
    )
    trace_path = _bound_repository_file(
        repository, evidence["trace"], "run01 trace"
    )
    high_motion = json.loads(high_motion_path.read_text(encoding="utf-8"))
    prior_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    first_stop = high_motion.get("wrist_ft", {}).get("first_safety_stop", {})
    if not (
        high_motion.get("abort_reason") == "WRIST_FT_RESULTANT_TORQUE_ABORT"
        and first_stop.get("reason") == "WRIST_FT_RESULTANT_TORQUE_ABORT"
        and first_stop.get("phase") == "approach_above"
        and float(first_stop.get("resultant_torque_nm", -1.0))
        > float(high_motion["wrist_ft"]["torque_limit_nm"])
        and high_motion.get("high_wait_reached") is False
        and int(high_motion.get("descent_command_count", -1)) == 0
        and int(high_motion.get("finger_closure_command_count", -1)) == 0
        and prior_trace.get("formal_dynamic_pass") is False
        and prior_trace.get("online_object_or_contact_truth_used") is False
    ):
        raise ValueError("run01 direct torque-stop evidence differs")
    override = document["single_variable_override"]
    if override != {
        "field": "controller.approach_high_motion_duration_s",
        "configured_value_s": 3.0,
        "effective_value_s": 6.0,
        "preshape_duration_s_unchanged": 3.0,
        "base_dynamic_mapping_unchanged": True,
        "minimum_jerk_peak_acceleration_ratio_vs_base": 0.25,
        "all_other_dynamic_fields_identical_to_base": True,
        "force_and_torque_limits_unchanged": True,
        "visual_target_path_grasp_and_static_panel_unchanged": True,
    }:
        raise ValueError("visual high-reobserve slow override differs")
    return base_path, base_document


def _load_visual_high_reobserve_slow2_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_visual_high_reobserve_slow_config",
        "direct_failure_evidence",
        "run02_direct_observation",
        "single_variable_override",
        "route_stop_rule",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_HIGH_REOBSERVE_SLOW2_SCHEMA
    ):
        raise ValueError("visual high-reobserve slow2 config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_FRESH_VISION_TO_APPROACH_HIGH_AND_REOBSERVE"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "approach_high_motion_authorized": True,
        "descent_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("visual high-reobserve slow2 authorization differs")
    slow_path = _bound_repository_file(
        repository,
        document["base_visual_high_reobserve_slow_config"],
        "base visual high-reobserve slow config",
    )
    slow_document = json.loads(slow_path.read_text(encoding="utf-8"))
    base_high_path, base_high_document = (
        _load_visual_high_reobserve_slow_contract(repository, slow_document)
    )
    evidence = document["direct_failure_evidence"]
    if set(evidence) != {"high_motion_execution", "trace"}:
        raise ValueError("visual high-reobserve slow2 evidence set differs")
    high_motion_path = _bound_repository_file(
        repository, evidence["high_motion_execution"], "run02 high motion"
    )
    trace_path = _bound_repository_file(
        repository, evidence["trace"], "run02 trace"
    )
    high_motion = json.loads(high_motion_path.read_text(encoding="utf-8"))
    prior_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    first_stop = high_motion.get("wrist_ft", {}).get("first_safety_stop", {})
    physics_dt_s = float(prior_trace.get("physics_dt_s", float("nan")))
    expected_total_steps = round(6.0 / physics_dt_s)
    observation = document["run02_direct_observation"]
    if observation != {
        "approach_above_completed_steps": 217,
        "approach_above_commanded_steps": 1440,
        "normalized_trajectory_fraction_u": 0.15069444444444444,
        "observed_stop_resultant_torque_nm": 0.05410430183424915,
        "unchanged_torque_soft_limit_nm": 0.0540690202,
        "theoretical_6p5s_peak_torque_extrapolation_nm": 0.0496134,
        "theoretical_margin_below_limit_nm": 0.00445562,
        "theoretical_values_are_preregistered_inference_not_dynamic_evidence": True,
    }:
        raise ValueError("visual high-reobserve slow2 run02 observation differs")
    if not (
        high_motion.get("abort_reason") == "WRIST_FT_RESULTANT_TORQUE_ABORT"
        and first_stop.get("reason") == "WRIST_FT_RESULTANT_TORQUE_ABORT"
        and first_stop.get("phase") == "approach_above"
        and int(high_motion.get("phase_counts", {}).get("approach_above", -1))
        == int(observation["approach_above_completed_steps"])
        and expected_total_steps
        == int(observation["approach_above_commanded_steps"])
        and math.isclose(
            217.0 / float(expected_total_steps),
            float(observation["normalized_trajectory_fraction_u"]),
            rel_tol=0.0,
            abs_tol=1.0e-16,
        )
        and math.isclose(
            float(first_stop.get("resultant_torque_nm", -1.0)),
            float(observation["observed_stop_resultant_torque_nm"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and float(first_stop["resultant_torque_nm"])
        > float(high_motion["wrist_ft"]["torque_limit_nm"])
        and math.isclose(
            float(observation["unchanged_torque_soft_limit_nm"]),
            float(high_motion["wrist_ft"]["torque_limit_nm"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            float(observation["unchanged_torque_soft_limit_nm"])
            - float(
                observation["theoretical_6p5s_peak_torque_extrapolation_nm"]
            ),
            float(observation["theoretical_margin_below_limit_nm"]),
            rel_tol=0.0,
            abs_tol=5.0e-10,
        )
        and high_motion.get("high_wait_reached") is False
        and int(high_motion.get("descent_command_count", -1)) == 0
        and int(high_motion.get("finger_closure_command_count", -1)) == 0
        and high_motion.get("approach_high_motion_duration", {}).get(
            "effective_s"
        )
        == 6.0
        and prior_trace.get("high_wait_reached") is False
        and prior_trace.get("second_provider_ready") is False
        and int(prior_trace.get("descent_command_count", -1)) == 0
        and int(prior_trace.get("finger_closure_command_count", -1)) == 0
        and prior_trace.get("formal_dynamic_pass") is False
        and prior_trace.get("online_object_or_contact_truth_used") is False
    ):
        raise ValueError("run02 direct torque-stop evidence differs")
    override = document["single_variable_override"]
    if override != {
        "field": "controller.approach_high_motion_duration_s",
        "configured_value_s": 6.0,
        "effective_value_s": 6.5,
        "preshape_duration_s_unchanged": 3.0,
        "base_dynamic_mapping_unchanged": True,
        "all_other_dynamic_fields_identical_to_slow_v1": True,
        "force_and_torque_limits_unchanged": True,
        "visual_target_path_grasp_and_static_panel_unchanged": True,
    }:
        raise ValueError("visual high-reobserve slow2 override differs")
    if document["route_stop_rule"] != {
        "same_earliest_physical_reason_without_new_information": (
            "WRIST_FT_RESULTANT_TORQUE_ABORT_DURING_APPROACH_ABOVE"
        ),
        "action_after_run03_repeat": "PARKED",
        "no_further_timing_variants": True,
    }:
        raise ValueError("visual high-reobserve slow2 stop rule differs")
    return base_high_path, base_high_document


def _load_visual_high_reobserve_dynamic_inertia_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_visual_high_reobserve_slow2_config",
        "sealed_free_space_evidence",
        "single_variable_override",
        "dynamic_inertia_contract",
        "offline_replay",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_HIGH_REOBSERVE_DYNAMIC_INERTIA_SCHEMA
    ):
        raise ValueError("visual high dynamic-inertia config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_FRESH_VISION_TO_APPROACH_HIGH_AND_REOBSERVE"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "approach_high_motion_authorized": True,
        "descent_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("visual high dynamic-inertia authorization differs")
    slow2_path = _bound_repository_file(
        repository,
        document["base_visual_high_reobserve_slow2_config"],
        "base visual high-reobserve slow2 config",
    )
    slow2_document = json.loads(slow2_path.read_text(encoding="utf-8"))
    base_high_path, base_high_document = (
        _load_visual_high_reobserve_slow2_contract(repository, slow2_document)
    )
    evidence = document["sealed_free_space_evidence"]
    if set(evidence) != {"run01", "run02", "run03"}:
        raise ValueError("dynamic-inertia evidence run set differs")
    expected_phases = {"run01": 28, "run02": 217, "run03": 291}
    for run_name, record in evidence.items():
        if set(record) != {"samples", "result"}:
            raise ValueError(f"dynamic-inertia {run_name} evidence differs")
        samples_path = _bound_repository_file(
            repository, record["samples"], f"{run_name} joint/FT samples"
        )
        result_path = _bound_repository_file(
            repository, record["result"], f"{run_name} high result"
        )
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not (
            samples.get(
                "online_object_semantic_instance_or_contact_truth_used"
            )
            is False
            and result.get("abort_reason")
            == "WRIST_FT_RESULTANT_TORQUE_ABORT"
            and int(result.get("phase_counts", {}).get("approach_above", -1))
            == expected_phases[run_name]
            and int(result.get("descent_command_count", -1)) == 0
            and int(result.get("finger_closure_command_count", -1)) == 0
        ):
            raise ValueError(f"dynamic-inertia {run_name} source result differs")
    if document["single_variable_override"] != {
        "field": "wrist_ft_safety.dynamic_inertia_compensation",
        "configured_value": "GRAVITY_ONLY",
        "effective_value": "GRAVITY_PLUS_FROZEN_HAND_DYNAMIC_INERTIA",
        "approach_high_motion_duration_s_unchanged": 6.5,
        "preshape_duration_s_unchanged": 3.0,
        "force_and_torque_limits_unchanged": True,
        "visual_target_path_grasp_and_static_panel_unchanged": True,
    }:
        raise ValueError("dynamic-inertia single-variable override differs")
    dynamic = document["dynamic_inertia_contract"]
    if not (
        dynamic.get("algorithm") == _CausalFrozenHandDynamicInertia.METHOD
        and dynamic.get("enabled_phases") == ["approach_above"]
        and int(dynamic.get("physics_rate_hz", -1)) == 240
        and dynamic.get("causal_history")
        == "CURRENT_AND_ONE_CONSECUTIVE_PREVIOUS_Q_QDOT_JACOBIAN"
        and dynamic.get("missing_or_nonconsecutive_history_action")
        == "FAIL_CLOSE_ABORT"
        and float(dynamic.get("joint_numerical_zero_clamp_rad", -1.0))
        == 1.0e-6
        and dynamic.get("physical_hand2arm_wrench_used_as_prediction_input")
        is False
        and dynamic.get("object_semantic_instance_or_contact_truth_used")
        is False
        and dynamic.get("online_parameter_fit_or_adaptation") is False
    ):
        raise ValueError("dynamic-inertia algorithm contract differs")
    _bound_repository_file(
        repository, dynamic["inertial_source"], "dynamic inertia source"
    )
    replay = document["offline_replay"]
    if not (
        replay.get("implementation_method_peak_force_torque")
        == {
            "run01": [0.001414773858820591, 0.000274504532825147],
            "run02": [0.0033259522454741875, 0.0008224391823176474],
            "run03": [0.004924746856995128, 0.0011572828314513673],
        }
        and replay.get("conservative_9_link_fk_sensitivity_peak_force_torque")
        == {
            "run01": [0.002533, 0.000887174],
            "run02": [0.033635, 0.010291962],
            "run03": [0.0275274, 0.00986958],
        }
        and replay.get("sensitivity_result_not_selected_to_reduce_reported_peak")
        is True
        and float(replay.get("maximum_resultant_force_n_unchanged", -1.0))
        == 3.0400615
        and float(replay.get("maximum_resultant_torque_nm_unchanged", -1.0))
        == 0.0540690202
        and replay.get("all_three_sealed_observed_segments_inside_unchanged_limits")
        is True
        and replay.get("full_unobserved_approach_path_proven") is False
    ):
        raise ValueError("dynamic-inertia offline replay contract differs")
    return base_high_path, base_high_document


def _load_visual_second_short_prefix_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_dynamic_inertia_config",
        "run05_evidence",
        "stationarity_gate_correction",
        "short_prefix",
        "runtime_static_gate",
        "offline_static_reference",
        "wrist_ft_safety",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_SECOND_SHORT_PREFIX_SCHEMA
    ):
        raise ValueError("visual second short-prefix config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_SECOND_RGBD_TO_SAFE_FREE_SPACE_PREFIX"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "second_visual_short_prefix_authorized": True,
        "descent_to_pregrasp_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("visual second short-prefix authorization differs")
    base_path = _bound_repository_file(
        repository,
        document["base_dynamic_inertia_config"],
        "second-prefix base dynamic-inertia config",
    )
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    if base_document.get("schema_version") != (
        VISION_HIGH_REOBSERVE_DYNAMIC_INERTIA_SCHEMA
    ):
        raise ValueError("second-prefix base is not dynamic-inertia high mode")

    evidence_paths = {
        name: _bound_repository_file(
            repository, record, f"second-prefix run05 {name}"
        )
        for name, record in document["run05_evidence"].items()
    }
    if set(evidence_paths) != {
        "high_motion_execution",
        "second_provider",
        "sealed_reobservation_report",
    }:
        raise ValueError("second-prefix run05 evidence set differs")
    high_motion = json.loads(
        evidence_paths["high_motion_execution"].read_text(encoding="utf-8")
    )
    provider = json.loads(
        evidence_paths["second_provider"].read_text(encoding="utf-8")
    )
    sealed = json.loads(
        evidence_paths["sealed_reobservation_report"].read_text(
            encoding="utf-8"
        )
    )
    correction = document["stationarity_gate_correction"]
    deltas = np.asarray(
        sealed.get("joint_position_change_during_paused_capture_rad"),
        dtype=np.float64,
    )
    run05_maximum_delta = (
        float(np.max(np.abs(deltas))) if deltas.ndim == 1 and len(deltas) else np.inf
    )
    if not (
        high_motion.get("high_wait_reached") is True
        and high_motion.get("abort_reason") is None
        and int(high_motion.get("descent_command_count", -1)) == 0
        and int(high_motion.get("finger_closure_command_count", -1)) == 0
        and provider.get("provider_scope") == "TRANSPORT_PLUG_ONLY"
        and provider.get("transport_grasp_pose", {}).get("status")
        == "OBSERVED_AXIS_POSITION_YAW_FREE"
        and provider.get("truth_flags", {}).get("uses_object_pose_truth") is False
        and sealed.get("provider_result_sealed_before_truth_read") is True
        and sealed.get("provider_image_gate_pass") is True
        and sealed.get("first_second_truth_free_consistency", {}).get("pass")
        is True
        and sealed.get("robot_api_counts_unchanged_during_capture_and_provider")
        is True
        and sealed.get("timeline_paused_for_capture") is True
        and sealed.get("second_provider_ready") is False
        and correction
        == {
            "method": "FLOAT32_READBACK_FOUR_EPSILON",
            "absolute_tolerance_rad": 4.76837158203125e-7,
            "run05_maximum_absolute_delta_rad": 2.384185791015625e-7,
            "run05_original_second_provider_ready": False,
            "run05_image_gate_pass": True,
            "run05_first_second_truth_free_consistency_pass": True,
            "run05_used_only_as_gate_and_offline_geometry_evidence": True,
            "run05_retroactive_online_control_claimed": False,
        }
        and math.isclose(
            run05_maximum_delta,
            float(correction["run05_maximum_absolute_delta_rad"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and run05_maximum_delta
        <= float(correction["absolute_tolerance_rad"])
    ):
        raise ValueError("second-prefix run05 direct evidence differs")

    prefix = document["short_prefix"]
    if prefix != {
        "target_waypoint_index": 1,
        "approach_high_clearance_m": 0.05,
        "target_clearance_from_pregrasp_m": 0.0375,
        "motion_command_count": 780,
        "runtime_state_count": 781,
        "motion_duration_s": 3.25,
        "settle_timeout_s": 1.0,
        "settle_error_rad": 0.02,
        "settle_consecutive_samples": 30,
        "motion_phase": "second_visual_short_path",
        "settle_phase": "second_visual_short_path_settle",
        "hand_command": "FROZEN_PREGRASP_OPENING",
        "trajectory_start": (
            "ACTUAL_ACTIVE_ARM_READBACK_AFTER_SECOND_PROVIDER_SEAL"
        ),
        "surface_displacement_reference": (
            "FIRST_PROVIDER_MOTION_PLAN_APPROACH_WAYPOINT_0"
        ),
        "second_provider_must_be_sealed_before_target_plan_or_command": True,
        "truth_read_allowed_only_after_target_plan_static_audit_and_execution_are_sealed": True,
    }:
        raise ValueError("visual second short-prefix motion contract differs")
    gate = document["runtime_static_gate"]
    if gate != {
        "method": (
            "COMPLETE_17_LINK_781_COMMAND_STATE_CONSERVATIVE_HIGH_CLEARANCE_PREFIX_V1"
        ),
        "collision_roster": (
            "src/kcg_connector/config/carts_collision_roster_v1.yaml"
        ),
        "collision_link_count": 17,
        "state_count": 781,
        "obstacle_clearance_lower_bound_minimum_m": 0.001,
        "nonadjacent_self_clearance_minimum_m": 0.0001,
        "joint_limit_margin_must_be_positive": True,
        "obstacle_lower_bound_rule": (
            "FROZEN_HIGH_MINIMUM_CLEARANCE_MINUS_MAXIMUM_LINK_SURFACE_"
            "DISPLACEMENT_FROM_FIRST_PLANNED_HIGH"
        ),
        "nonadjacent_self_collision_method": (
            "FCL_DISTANCE_ON_ACTUAL_START_AND_EVERY_RUNTIME_PREFIX_COMMAND_"
            "TARGET_STATE"
        ),
        "continuous_joint_space_proof_claimed": False,
    }:
        raise ValueError("visual second short-prefix static gate differs")
    roster_path = (repository / str(gate["collision_roster"])).resolve()
    if not roster_path.is_file():
        raise ValueError("visual second short-prefix collision roster is missing")

    offline = document["offline_static_reference"]
    expected_offline = {
        "scope": "RUN05_GEOMETRY_REHEARSAL_NOT_RUNTIME_AUTHORIZATION",
        "maximum_link_surface_displacement_from_first_planned_high_m": (
            0.014181229110475667
        ),
        "conservative_table_clearance_lower_bound_m": 0.01030897236276044,
        "conservative_range_robust_plug_clearance_lower_bound_m": (
            0.00883304848256012
        ),
        "conservative_fixture_clearance_lower_bound_m": 0.28064389056828076,
        "minimum_nonadjacent_self_clearance_m": 0.0005531182686928471,
        "minimum_arm_joint_limit_margin_rad": 0.3034925346272648,
        "all_runtime_gate_thresholds_satisfied": True,
        "runtime_reaudit_still_required": True,
    }
    if offline != expected_offline:
        raise ValueError("visual second short-prefix offline reference differs")
    displacement = float(
        offline["maximum_link_surface_displacement_from_first_planned_high_m"]
    )
    base_high_path, base_high = _load_visual_high_reobserve_dynamic_inertia_contract(
        repository, base_document
    )
    high_contract = json.loads(base_high_path.read_text(encoding="utf-8"))
    high_panel = high_contract["static_path_panel"]
    derived = (
        float(high_panel["minimum_table_clearance_m"]) - displacement,
        float(high_panel["minimum_range_robust_plug_clearance_m"])
        - displacement,
        float(high_panel["minimum_fixture_clearance_m"]) - displacement,
    )
    expected_derived = (
        float(offline["conservative_table_clearance_lower_bound_m"]),
        float(
            offline[
                "conservative_range_robust_plug_clearance_lower_bound_m"
            ]
        ),
        float(offline["conservative_fixture_clearance_lower_bound_m"]),
    )
    if not np.allclose(derived, expected_derived, rtol=0.0, atol=1.0e-15):
        raise ValueError("second-prefix conservative clearance arithmetic differs")
    if document["wrist_ft_safety"] != {
        "physical_source_joint": "hand2arm",
        "maximum_resultant_force_n": 3.0400615,
        "maximum_resultant_torque_nm": 0.0540690202,
        "dynamic_inertia_enabled_phases": [
            "approach_above",
            "second_visual_short_path",
            "second_visual_short_path_settle",
        ],
        "threshold_action": "STOP_AND_HOLD_NO_FURTHER_MOTION",
        "limits_unchanged_from_base": True,
    }:
        raise ValueError("visual second short-prefix FT safety differs")
    return base_path, base_document


def _load_visual_second_short_prefix_exact_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_short_prefix_config",
        "run06_evidence",
        "short_prefix",
        "runtime_static_gate",
        "offline_static_reference",
        "wrist_ft_safety",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_SECOND_SHORT_PREFIX_EXACT_SCHEMA
    ):
        raise ValueError("exact visual short-prefix config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_SECOND_RGBD_TO_EXACT_SAFE_FREE_SPACE_PREFIX"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "second_visual_short_prefix_authorized": True,
        "descent_to_pregrasp_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("exact visual short-prefix authorization differs")
    base_path = _bound_repository_file(
        repository,
        document["base_short_prefix_config"],
        "exact-prefix base short-prefix config",
    )
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    if base_document.get("schema_version") != VISION_SECOND_SHORT_PREFIX_SCHEMA:
        raise ValueError("exact-prefix base is not the 37.5 mm short-prefix")
    dynamic_path, dynamic_document = _load_visual_second_short_prefix_contract(
        repository, base_document
    )

    evidence_paths = {
        name: _bound_repository_file(
            repository, record, f"exact-prefix run06 {name}"
        )
        for name, record in document["run06_evidence"].items()
    }
    if set(evidence_paths) != {"result", "static_audit", "execution"}:
        raise ValueError("exact-prefix run06 evidence set differs")
    result = json.loads(evidence_paths["result"].read_text(encoding="utf-8"))
    static = json.loads(
        evidence_paths["static_audit"].read_text(encoding="utf-8")
    )
    execution = json.loads(
        evidence_paths["execution"].read_text(encoding="utf-8")
    )
    if not (
        result.get("status") == "DYNAMIC_PASS"
        and result.get("second_provider_ready") is True
        and result.get("second_visual_short_prefix_success") is True
        and int(result.get("free_space_descent_command_count", -1)) == 780
        and int(
            result.get("descent_to_pregrasp_or_contact_command_count", -1)
        )
        == 0
        and int(result.get("finger_closure_command_count", -1)) == 0
        and static.get("pass") is True
        and int(static.get("state_count", -1)) == 781
        and static.get("first_self_collision") is None
        and execution.get("success") is True
        and execution.get("abort_reason") is None
        and execution.get("wrist_ft", {}).get("first_safety_stop") is None
        and execution.get("object_contact_authorized") is False
        and execution.get(
            "online_object_semantic_instance_or_contact_truth_used"
        )
        is False
        and execution.get("control_trajectory_sealed_before_truth_read") is True
    ):
        raise ValueError("exact-prefix run06 direct evidence differs")

    prefix = document["short_prefix"]
    if prefix != {
        "target_method": "BOUNDED_IK_EXACT_CARTESIAN_CLEARANCE",
        "ik_seed_waypoint_indices": [1, 2],
        "approach_high_clearance_m": 0.05,
        "target_clearance_from_pregrasp_m": 0.03125,
        "motion_command_count": 960,
        "runtime_state_count": 961,
        "motion_duration_s": 4.0,
        "settle_timeout_s": 1.0,
        "settle_error_rad": 0.02,
        "settle_consecutive_samples": 30,
        "motion_phase": "second_visual_short_path",
        "settle_phase": "second_visual_short_path_settle",
        "hand_command": "FROZEN_PREGRASP_OPENING",
        "trajectory_start": (
            "ACTUAL_ACTIVE_ARM_READBACK_AFTER_SECOND_PROVIDER_SEAL"
        ),
        "surface_displacement_reference": (
            "FIRST_PROVIDER_MOTION_PLAN_APPROACH_WAYPOINT_0"
        ),
        "second_provider_must_be_sealed_before_target_plan_or_command": True,
        "truth_read_allowed_only_after_target_plan_static_audit_and_execution_are_sealed": True,
    }:
        raise ValueError("exact visual short-prefix motion contract differs")
    gate = document["runtime_static_gate"]
    if gate != {
        "method": (
            "COMPLETE_17_LINK_961_COMMAND_STATE_CONSERVATIVE_HIGH_CLEARANCE_"
            "PREFIX_V1"
        ),
        "collision_roster": (
            "src/kcg_connector/config/carts_collision_roster_v1.yaml"
        ),
        "collision_link_count": 17,
        "state_count": 961,
        "obstacle_clearance_lower_bound_minimum_m": 0.001,
        "nonadjacent_self_clearance_minimum_m": 0.0001,
        "joint_limit_margin_must_be_positive": True,
        "obstacle_lower_bound_rule": (
            "FROZEN_HIGH_MINIMUM_CLEARANCE_MINUS_MAXIMUM_LINK_SURFACE_"
            "DISPLACEMENT_FROM_FIRST_PLANNED_HIGH"
        ),
        "nonadjacent_self_collision_method": (
            "FCL_DISTANCE_ON_ACTUAL_START_AND_EVERY_RUNTIME_PREFIX_COMMAND_"
            "TARGET_STATE"
        ),
        "continuous_joint_space_proof_claimed": False,
    }:
        raise ValueError("exact visual short-prefix static gate differs")
    offline = document["offline_static_reference"]
    expected_offline = {
        "scope": (
            "RUN06_GEOMETRY_REHEARSAL_WITH_EXACT_31P25MM_IK_NOT_RUNTIME_"
            "AUTHORIZATION"
        ),
        "ik_position_error_m": 1.1443916996305594e-16,
        "ik_rotation_error_rad": 1.860289300072345e-16,
        "target_arm_rad": [
            2.1912823222789624,
            -0.5474459176213531,
            -2.114334529223118,
            -1.0734898737796732,
            -0.4691577007342628,
            1.7423950124556586,
            -1.1289101739069805,
        ],
        "maximum_link_surface_displacement_from_first_planned_high_m": (
            0.021255339017633927
        ),
        "conservative_table_clearance_lower_bound_m": 0.003234862455602179,
        "conservative_range_robust_plug_clearance_lower_bound_m": (
            0.0017589385754018597
        ),
        "conservative_fixture_clearance_lower_bound_m": 0.2735697806611225,
        "minimum_nonadjacent_self_clearance_m": 0.0005531182686927304,
        "minimum_active_joint_limit_margin_rad": 0.06943959712982184,
        "all_runtime_gate_thresholds_satisfied": True,
        "runtime_reaudit_still_required": True,
    }
    if offline != expected_offline:
        raise ValueError("exact visual short-prefix offline reference differs")
    base_high_path, _ = _load_visual_high_reobserve_dynamic_inertia_contract(
        repository, dynamic_document
    )
    high = json.loads(base_high_path.read_text(encoding="utf-8"))
    displacement = float(
        offline["maximum_link_surface_displacement_from_first_planned_high_m"]
    )
    derived = (
        float(high["static_path_panel"]["minimum_table_clearance_m"])
        - displacement,
        float(
            high["static_path_panel"][
                "minimum_range_robust_plug_clearance_m"
            ]
        )
        - displacement,
        float(high["static_path_panel"]["minimum_fixture_clearance_m"])
        - displacement,
    )
    expected_derived = (
        float(offline["conservative_table_clearance_lower_bound_m"]),
        float(
            offline["conservative_range_robust_plug_clearance_lower_bound_m"]
        ),
        float(offline["conservative_fixture_clearance_lower_bound_m"]),
    )
    if not np.allclose(derived, expected_derived, rtol=0.0, atol=1.0e-15):
        raise ValueError("exact-prefix conservative clearance arithmetic differs")
    if document["wrist_ft_safety"] != base_document["wrist_ft_safety"]:
        raise ValueError("exact-prefix FT safety differs from run06")
    return dynamic_path, dynamic_document


def _load_visual_second_short_prefix_exact_obstacles_contract(
    repository: Path,
    document: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "base_short_prefix_config",
        "run07_evidence",
        "short_prefix",
        "runtime_static_gate",
        "offline_static_reference",
        "wrist_ft_safety",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        VISION_SECOND_SHORT_PREFIX_EXACT_OBSTACLES_SCHEMA
    ):
        raise ValueError("exact-obstacle prefix config schema/fields differ")
    if document.get("mode") != (
        "SIMULATION_ONLY_SECOND_RGBD_TO_EXACT_OBSTACLE_GATED_PREFIX"
    ) or document.get("authorization") != {
        "hardware_authorized": False,
        "simulation_only": True,
        "second_visual_short_prefix_authorized": True,
        "descent_to_pregrasp_authorized": False,
        "object_contact_authorized": False,
        "finger_closure_authorized": False,
    }:
        raise ValueError("exact-obstacle prefix authorization differs")
    base_path = _bound_repository_file(
        repository,
        document["base_short_prefix_config"],
        "exact-obstacle base short-prefix config",
    )
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    if base_document.get("schema_version") != (
        VISION_SECOND_SHORT_PREFIX_EXACT_SCHEMA
    ):
        raise ValueError("exact-obstacle base is not the 31.25 mm prefix")
    dynamic_path, dynamic_document = (
        _load_visual_second_short_prefix_exact_contract(
            repository, base_document
        )
    )
    evidence_paths = {
        name: _bound_repository_file(
            repository, record, f"exact-obstacle run07 {name}"
        )
        for name, record in document["run07_evidence"].items()
    }
    if set(evidence_paths) != {"result", "static_audit", "execution"}:
        raise ValueError("exact-obstacle run07 evidence set differs")
    result = json.loads(evidence_paths["result"].read_text(encoding="utf-8"))
    static = json.loads(
        evidence_paths["static_audit"].read_text(encoding="utf-8")
    )
    execution = json.loads(
        evidence_paths["execution"].read_text(encoding="utf-8")
    )
    if not (
        result.get("status") == "DYNAMIC_PASS"
        and float(result.get("target_clearance_from_pregrasp_m", -1.0))
        == 0.03125
        and result.get("target_method")
        == "BOUNDED_IK_EXACT_CARTESIAN_CLEARANCE"
        and result.get("second_visual_short_prefix_success") is True
        and int(result.get("free_space_descent_command_count", -1)) == 960
        and int(
            result.get("descent_to_pregrasp_or_contact_command_count", -1)
        )
        == 0
        and static.get("pass") is True
        and int(static.get("state_count", -1)) == 961
        and execution.get("success") is True
        and execution.get("abort_reason") is None
        and execution.get("wrist_ft", {}).get("first_safety_stop") is None
        and execution.get("object_contact_authorized") is False
        and execution.get(
            "online_object_semantic_instance_or_contact_truth_used"
        )
        is False
    ):
        raise ValueError("exact-obstacle run07 direct evidence differs")
    prefix = document["short_prefix"]
    if prefix != {
        "target_method": "PLANNER_WAYPOINT_INDEX",
        "target_waypoint_index": 2,
        "approach_high_clearance_m": 0.05,
        "target_clearance_from_pregrasp_m": 0.025,
        "motion_command_count": 1104,
        "runtime_state_count": 1105,
        "motion_duration_s": 4.6,
        "settle_timeout_s": 1.0,
        "settle_error_rad": 0.02,
        "settle_consecutive_samples": 30,
        "motion_phase": "second_visual_short_path",
        "settle_phase": "second_visual_short_path_settle",
        "hand_command": "FROZEN_PREGRASP_OPENING",
        "trajectory_start": (
            "ACTUAL_ACTIVE_ARM_READBACK_AFTER_SECOND_PROVIDER_SEAL"
        ),
        "surface_displacement_reference": (
            "FIRST_PROVIDER_MOTION_PLAN_APPROACH_WAYPOINT_0"
        ),
        "second_provider_must_be_sealed_before_target_plan_or_command": True,
        "truth_read_allowed_only_after_target_plan_static_audit_and_execution_are_sealed": True,
    }:
        raise ValueError("exact-obstacle prefix motion contract differs")
    gate = document["runtime_static_gate"]
    if gate != {
        "method": (
            "COMPLETE_17_LINK_1105_COMMAND_STATE_EXACT_OBSTACLE_PREFIX_V1"
        ),
        "collision_roster": (
            "src/kcg_connector/config/carts_collision_roster_v1.yaml"
        ),
        "collision_link_count": 17,
        "state_count": 1105,
        "obstacle_clearance_lower_bound_minimum_m": 0.001,
        "nonadjacent_self_clearance_minimum_m": 0.0001,
        "joint_limit_margin_must_be_positive": True,
        "obstacle_lower_bound_rule": (
            "EXACT_TABLE_AND_YAW_SWEPT_CAD_CYLINDER_FCL_MINUS_DERIVED_POSE_"
            "ENVELOPE"
        ),
        "yaw_swept_cad_cylinder": {
            "registration_cad": {
                "path": (
                    "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
                    "D38999_26FJ35PN_VISUAL.stl"
                ),
                "sha256": (
                    "89bbf4ba3b8b5fb388276a378a820bb7547e676e97f0b9e2635b7862a86aa617"
                ),
            },
            "local_axis": "+Z",
            "radius_m": 0.02367280046246563,
            "z_min_m": -0.031013399124145507,
            "z_max_m": 0.0,
            "origin_radius_m": 0.03901579689009648,
            "yaw_range_rad": [-math.pi, math.pi],
            "yaw_coverage": (
                "ALL_CAD_LOCAL_Z_ROTATIONS_CONTAINED_BY_CYLINDER"
            ),
        },
        "pose_envelope_derivation": {
            "lateral_x_absolute_m": 0.000183,
            "lateral_y_absolute_m": 0.000183,
            "support_z_absolute_m": 0.003,
            "axis_tilt_cone_rad": 0.08726646259971647,
            "translation_norm_bound_m": 0.0030111423081614725,
            "tilt_surface_displacement_bound_m": 0.00340369031583278,
            "derived_surface_displacement_envelope_m": 0.006414832623994253,
            "formula": "NORM_XYZ_TRANSLATION_PLUS_2R_SIN_TILT_OVER_2",
        },
        "nonadjacent_self_collision_method": (
            "FCL_DISTANCE_ON_ACTUAL_START_AND_EVERY_RUNTIME_PREFIX_COMMAND_"
            "TARGET_STATE"
        ),
        "continuous_joint_space_proof_claimed": False,
    }:
        raise ValueError("exact-obstacle prefix static gate differs")
    _bound_repository_file(
        repository,
        gate["yaw_swept_cad_cylinder"]["registration_cad"],
        "exact-obstacle yaw-swept registration CAD",
    )
    cylinder = gate["yaw_swept_cad_cylinder"]
    envelope = gate["pose_envelope_derivation"]
    translation_bound = math.sqrt(
        float(envelope["lateral_x_absolute_m"]) ** 2
        + float(envelope["lateral_y_absolute_m"]) ** 2
        + float(envelope["support_z_absolute_m"]) ** 2
    )
    tilt_bound = 2.0 * float(cylinder["origin_radius_m"]) * math.sin(
        0.5 * float(envelope["axis_tilt_cone_rad"])
    )
    if not (
        math.isclose(
            translation_bound,
            float(envelope["translation_norm_bound_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            tilt_bound,
            float(envelope["tilt_surface_displacement_bound_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            translation_bound + tilt_bound,
            float(envelope["derived_surface_displacement_envelope_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("exact-obstacle pose envelope derivation differs")
    offline = document["offline_static_reference"]
    expected_offline = {
        "scope": (
            "RUN07_FULL_1105_STATE_EXACT_OBSTACLE_REHEARSAL_NOT_RUNTIME_"
            "AUTHORIZATION"
        ),
        "table_min_distance_m": 0.024490201473236106,
        "table_limiting_link": "iiwa_link_0",
        "table_limiting_state": 0,
        "yaw_swept_cylinder_min_distance_m": 0.0076564824328170826,
        "cylinder_limiting_link": "f2Link2",
        "cylinder_limiting_state": 1104,
        "derived_pose_surface_displacement_envelope_m": (
            0.006414832623994253
        ),
        "range_robust_yaw_swept_lower_bound_m": 0.00124164980882283,
        "fixture_conservative_lower_bound_m": 0.26650635435049147,
        "maximum_link_surface_displacement_m": 0.02831876532826495,
        "minimum_nonadjacent_self_clearance_m": 0.0005531182686927393,
        "minimum_active_joint_limit_margin_rad": 0.06943959712982184,
        "collision_count": 0,
        "all_runtime_gate_thresholds_satisfied": True,
        "runtime_reaudit_still_required": True,
    }
    if offline != expected_offline:
        raise ValueError("exact-obstacle prefix offline reference differs")
    if not math.isclose(
        float(offline["yaw_swept_cylinder_min_distance_m"])
        - float(offline["derived_pose_surface_displacement_envelope_m"]),
        float(offline["range_robust_yaw_swept_lower_bound_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("exact-obstacle robust object arithmetic differs")
    if document["wrist_ft_safety"] != base_document["wrist_ft_safety"]:
        raise ValueError("exact-obstacle FT safety differs from run07")
    return dynamic_path, dynamic_document


def _load_same_reset_rgbd_config(
    repository: Path,
    arguments: argparse.Namespace,
    inputs,
    scene_entry: Mapping[str, object],
) -> None:
    path = Path(arguments.same_reset_rgbd_config).expanduser().resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError("same-reset RGB-D config escapes the repository") from error
    document = json.loads(path.read_text(encoding="utf-8"))
    prefix_path = None
    prefix_document = None
    if document.get("schema_version") == VISION_SECOND_SHORT_PREFIX_SCHEMA:
        if arguments.mode != VISION_HIGH_REOBSERVE_MODE:
            raise ValueError(
                "visual second short-prefix config requires high-reobserve mode"
            )
        prefix_path = path
        prefix_document = document
        path, document = _load_visual_second_short_prefix_contract(
            repository, prefix_document
        )
    elif document.get("schema_version") == (
        VISION_SECOND_SHORT_PREFIX_EXACT_SCHEMA
    ):
        if arguments.mode != VISION_HIGH_REOBSERVE_MODE:
            raise ValueError(
                "exact visual second short-prefix config requires high-reobserve mode"
            )
        prefix_path = path
        prefix_document = document
        path, document = _load_visual_second_short_prefix_exact_contract(
            repository, prefix_document
        )
    elif document.get("schema_version") == (
        VISION_SECOND_SHORT_PREFIX_EXACT_OBSTACLES_SCHEMA
    ):
        if arguments.mode != VISION_HIGH_REOBSERVE_MODE:
            raise ValueError(
                "exact-obstacle visual short-prefix config requires high-reobserve mode"
            )
        prefix_path = path
        prefix_document = document
        path, document = (
            _load_visual_second_short_prefix_exact_obstacles_contract(
                repository, prefix_document
            )
        )
    arguments.visual_high_reobserve_config_path = None
    arguments.visual_high_reobserve_document = None
    arguments.visual_high_reobserve_variant_document = None
    arguments.visual_second_short_prefix_config_path = prefix_path
    arguments.visual_second_short_prefix_document = prefix_document
    arguments.dynamic_inertia_compensation_enabled = False
    arguments.dynamic_inertia_contract = None
    arguments.dynamic_inertia_offline_replay_audit = None
    arguments.approach_high_motion_duration_configured_s = None
    arguments.approach_high_motion_duration_effective_s = None
    if document.get("schema_version") == VISION_HIGH_REOBSERVE_SCHEMA:
        if arguments.mode not in VISION_MOTION_MODES:
            raise ValueError("high-reobserve config requires its dedicated mode")
        arguments.visual_high_reobserve_config_path = path
        arguments.visual_high_reobserve_document = document
        path, document = _load_visual_high_reobserve_contract(
            repository, path, document
        )
        configured = float(
            arguments.dynamic_settings["approach_above_duration_s"]
        )
        arguments.approach_high_motion_duration_configured_s = configured
        arguments.approach_high_motion_duration_effective_s = configured
    elif document.get("schema_version") == VISION_HIGH_REOBSERVE_SLOW_SCHEMA:
        if arguments.mode != VISION_HIGH_REOBSERVE_MODE:
            raise ValueError("slow high-reobserve config requires its dedicated mode")
        variant_path = path
        variant_document = document
        base_high_path, base_high_document = (
            _load_visual_high_reobserve_slow_contract(
                repository, variant_document
            )
        )
        path, document = _load_visual_high_reobserve_contract(
            repository, base_high_path, base_high_document
        )
        configured = float(
            arguments.dynamic_settings["approach_above_duration_s"]
        )
        override = variant_document["single_variable_override"]
        if configured != float(override["configured_value_s"]):
            raise ValueError("configured approach-high duration differs from slow base")
        arguments.visual_high_reobserve_config_path = variant_path
        arguments.visual_high_reobserve_document = base_high_document
        arguments.visual_high_reobserve_variant_document = variant_document
        arguments.approach_high_motion_duration_configured_s = configured
        arguments.approach_high_motion_duration_effective_s = float(
            override["effective_value_s"]
        )
    elif document.get("schema_version") == VISION_HIGH_REOBSERVE_SLOW2_SCHEMA:
        if arguments.mode != VISION_HIGH_REOBSERVE_MODE:
            raise ValueError("slow2 high-reobserve config requires its dedicated mode")
        variant_path = path
        variant_document = document
        base_high_path, base_high_document = (
            _load_visual_high_reobserve_slow2_contract(
                repository, variant_document
            )
        )
        path, document = _load_visual_high_reobserve_contract(
            repository, base_high_path, base_high_document
        )
        dynamic_duration = float(
            arguments.dynamic_settings["approach_above_duration_s"]
        )
        override = variant_document["single_variable_override"]
        if dynamic_duration != float(override["preshape_duration_s_unchanged"]):
            raise ValueError("slow2 preshape duration differs from the frozen base")
        arguments.visual_high_reobserve_config_path = variant_path
        arguments.visual_high_reobserve_document = base_high_document
        arguments.visual_high_reobserve_variant_document = variant_document
        arguments.approach_high_motion_duration_configured_s = float(
            override["configured_value_s"]
        )
        arguments.approach_high_motion_duration_effective_s = float(
            override["effective_value_s"]
        )
    elif document.get("schema_version") == (
        VISION_HIGH_REOBSERVE_DYNAMIC_INERTIA_SCHEMA
    ):
        if arguments.mode not in VISION_MOTION_MODES:
            raise ValueError(
                "dynamic-inertia high-reobserve config requires its dedicated mode"
            )
        variant_path = path
        variant_document = document
        base_high_path, base_high_document = (
            _load_visual_high_reobserve_dynamic_inertia_contract(
                repository, variant_document
            )
        )
        path, document = _load_visual_high_reobserve_contract(
            repository, base_high_path, base_high_document
        )
        dynamic_contract = variant_document["dynamic_inertia_contract"]
        if (
            float(arguments.dynamic_settings["approach_above_duration_s"])
            != float(
                variant_document["single_variable_override"][
                    "preshape_duration_s_unchanged"
                ]
            )
            or not math.isclose(
                1.0 / float(arguments.dynamic_settings["physics_dt_s"]),
                float(dynamic_contract["physics_rate_hz"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("dynamic-inertia timing differs from the frozen base")
        inertial_path = _bound_repository_file(
            repository,
            dynamic_contract["inertial_source"],
            "dynamic inertia source",
        )
        inertials = _load_frozen_hand_inertials(inertial_path)
        aggregate = _aggregate_frozen_hand_inertia(
            inputs.robot_model,
            inertials,
            dynamic_contract["frozen_pregrasp_hand_positions_rad"],
        )
        if not (
            math.isclose(
                float(aggregate["mass_kg"]),
                float(dynamic_contract["aggregate_mass_kg"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and np.allclose(
                aggregate["center_of_mass_sensor_m"],
                dynamic_contract["aggregate_center_of_mass_sensor_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                aggregate["inertia_about_com_sensor_kg_m2"],
                dynamic_contract[
                    "aggregate_inertia_about_com_sensor_kg_m2"
                ],
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("dynamic-inertia aggregate identity differs")
        replay_audit: dict[str, Any] = {}
        expected_replay = variant_document["offline_replay"][
            "implementation_method_peak_force_torque"
        ]
        for run_name, evidence_record in variant_document[
            "sealed_free_space_evidence"
        ].items():
            sample_path = _bound_repository_file(
                repository,
                evidence_record["samples"],
                f"{run_name} dynamic-inertia replay samples",
            )
            sample_document = json.loads(
                sample_path.read_text(encoding="utf-8")
            )
            replay_result = replay_high_observation_dynamic_inertia(
                samples=sample_document["samples"],
                robot_model=inputs.robot_model,
                hand_inertials=inertials,
                frozen_hand_positions=dynamic_contract[
                    "frozen_pregrasp_hand_positions_rad"
                ],
                physics_dt_s=float(
                    arguments.dynamic_settings["physics_dt_s"]
                ),
            )
            expected_peak = np.asarray(
                expected_replay[run_name], dtype=np.float64
            )
            actual_peak = np.asarray(
                (
                    replay_result["new_enabled_phase_peak_force_n"],
                    replay_result["new_enabled_phase_peak_torque_nm"],
                ),
                dtype=np.float64,
            )
            if not (
                replay_result["algorithm"] == dynamic_contract["algorithm"]
                and replay_result["enabled_phase_not_ready_count"] == 0
                and np.allclose(
                    actual_peak, expected_peak, rtol=0.0, atol=1.0e-12
                )
                and replay_result["new_full_record_peak_force_n"]
                <= float(
                    variant_document["offline_replay"][
                        "maximum_resultant_force_n_unchanged"
                    ]
                )
                and replay_result["new_full_record_peak_torque_nm"]
                <= float(
                    variant_document["offline_replay"][
                        "maximum_resultant_torque_nm_unchanged"
                    ]
                )
                and replay_result[
                    "physical_wrench_used_as_prediction_input"
                ]
                is False
                and replay_result[
                    "object_semantic_instance_or_contact_truth_used"
                ]
                is False
            ):
                raise ValueError(
                    f"{run_name} dynamic-inertia implementation replay differs"
                )
            replay_audit[run_name] = replay_result
        arguments.visual_high_reobserve_config_path = variant_path
        arguments.visual_high_reobserve_document = base_high_document
        arguments.visual_high_reobserve_variant_document = variant_document
        arguments.dynamic_inertia_compensation_enabled = True
        arguments.dynamic_inertia_contract = dynamic_contract
        arguments.dynamic_inertia_offline_replay_audit = replay_audit
        arguments.approach_high_motion_duration_configured_s = 6.5
        arguments.approach_high_motion_duration_effective_s = 6.5
    elif arguments.mode in VISION_MOTION_MODES:
        raise ValueError("visual high-reobserve mode requires its versioned config")
    if arguments.mode == VISION_GRASP_SERVO_MODE and (
        arguments.dynamic_inertia_compensation_enabled is not True
        or arguments.visual_second_short_prefix_document is not None
    ):
        raise ValueError(
            "vision-grasp-servo requires the existing dynamic-inertia high config "
            "without a distance-prefix config"
        )
    expected_top = {
        "schema_version",
        "mode",
        "authorization",
        "physical_scene_binding",
        "frozen_sources",
        "timing",
        "image_gates",
        "posthoc_research_gates",
        "truth_firewall",
        "experiment_contract",
    }
    if set(document) != expected_top or document.get("schema_version") != (
        SAME_RESET_RGBD_SCHEMA
    ):
        raise ValueError("same-reset RGB-D config schema/fields differ")
    if prefix_path is not None:
        arguments.visual_high_reobserve_config_path = prefix_path
    if document.get("mode") != (
        "SIMULATION_ONLY_SAME_RESET_RGBD_OBSERVE_ZERO_ROBOT_COMMANDS"
    ):
        raise ValueError("same-reset RGB-D mode contract differs")
    authorization = document["authorization"]
    if authorization != {
        "simulation_only": True,
        "hardware_authorized": False,
        "robot_motion_authorized": False,
        "visual_control_authorized": False,
        "receptacle_contact_authorized": False,
    }:
        raise ValueError("same-reset RGB-D authorization is not fail-closed")
    binding = document["physical_scene_binding"]
    if (
        binding.get("object_id") != arguments.object_id
        or binding.get("required_scene_kind")
        != "FREE_SINGLE_RIGID_ON_SHARED_FINITE_TABLE"
        or scene_entry.get("scene_kind") != binding.get("required_scene_kind")
        or file_sha256(inputs.config.path) != binding.get("grasp_config_sha256")
    ):
        raise ValueError("same-reset RGB-D physical scene binding differs")
    configured_path = (repository / binding["grasp_config"]).resolve()
    if configured_path != inputs.config.path:
        raise ValueError("same-reset RGB-D grasp config path differs")

    sources = document["frozen_sources"]
    source_pairs = (
        ("near_side_camera_config", "near_side_camera_config_sha256"),
        ("rgbd_bootstrap_config", "rgbd_bootstrap_config_sha256"),
        ("provider_input_template", "provider_input_template_sha256"),
        ("static_background_depth", "static_background_depth_sha256"),
    )
    resolved = {}
    for path_key, sha_key in source_pairs:
        source = (repository / sources[path_key]).resolve()
        try:
            source.relative_to(repository)
        except ValueError as error:
            raise ValueError(f"same-reset source {path_key} escapes repository") from error
        if not source.is_file() or file_sha256(source) != sources[sha_key]:
            raise ValueError(f"same-reset source {path_key} identity differs")
        resolved[path_key] = source
    import yaml

    camera_document = yaml.safe_load(
        resolved["near_side_camera_config"].read_text(encoding="utf-8")
    )
    camera = camera_document.get("camera", {})
    if (
        camera_document.get("schema_version") != "kcg_te_rgbd_camera_v1"
        or camera.get("calibration_id") != "te_near_side_rgbd_sim_v1"
        or camera.get("channels_exactly") != ["rgb", "distance_to_image_plane"]
        or camera.get("resolution_px") != [1280, 720]
    ):
        raise ValueError("same-reset near-side camera contract differs")
    static_depth = np.load(
        resolved["static_background_depth"], allow_pickle=False
    )
    if static_depth.shape != (720, 1280):
        raise ValueError("same-reset frozen background shape differs")

    timing = document["timing"]
    expected_timing = {
        "settle_duration_s": 2.0,
        "tail_observation_duration_s": 2.0,
        "capture_rt_subframes": 4,
        "maximum_frame_to_provider_seal_s": 30.0,
        "required_reset_count": 1,
    }
    if timing != expected_timing:
        raise ValueError("same-reset timing contract differs")
    gates = document["posthoc_research_gates"]
    if gates != {
        "maximum_absolute_support_z_error_m": 0.003,
        "maximum_axis_tilt_rad": 0.08726646259971647,
        "maximum_tail_translation_m": 0.001,
    }:
        raise ValueError("same-reset posthoc gates differ")
    image_gates = document["image_gates"]
    if image_gates != {
        "minimum_foreground_depth_delta_m": 0.00025,
        "minimum_plug_foreground_points": 150,
        "minimum_rx180_ring_points": 1000,
        "minimum_ring_radius_m": 0.021,
        "maximum_ring_radius_m": 0.0235,
        "maximum_ring_absolute_residual_p99_m": 0.0001,
    }:
        raise ValueError("same-reset image gates differ")
    firewall = document["truth_firewall"]
    if (
        firewall.get("provider_scope") != "TRANSPORT_PLUG_ONLY"
        or firewall.get("posthoc_truth_only_after_provider_file_sealed") is not True
        or not {
            "semantic_segmentation_truth",
            "instance_segmentation_truth",
            "object_pose_truth",
            "object_prim_path",
            "renderer_truth_point_cloud",
            "contact_truth",
        }.issubset(set(firewall.get("forbidden_provider_inputs", ())))
    ):
        raise ValueError("same-reset truth firewall differs")
    if not arguments.capture_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("same-reset capture ID is not a stable identifier")
    arguments.same_reset_rgbd_config_path = path
    arguments.same_reset_rgbd_document = document
    arguments.same_reset_rgbd_sources = resolved
    arguments.same_reset_rgbd_camera_document = camera_document


def _arguments(repository: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "isolated-hand",
            "preflight",
            "first-finger-diagnostic",
            "grasp-lift",
            SAME_RESET_RGBD_MODE,
            VISION_HIGH_REOBSERVE_MODE,
            VISION_GRASP_SERVO_MODE,
        ),
        required=True,
    )
    parser.add_argument("--object-id", default="current_d38999_26kj61sn_public_spec")
    parser.add_argument("--config", default=str(
        repository / "src/kcg_connector/config/carts_grasp_v2.yaml"))
    parser.add_argument("--runtime-resources", default=str(
        repository / "src/kcg_connector/config/carts_v2_isaac_runtime.json"))
    parser.add_argument(
        "--robot-asset",
        help="explicit simulation robot asset; defaults to dynamic.robot_asset",
    )
    parser.add_argument(
        "--free-split-object-manifest",
        help=(
            "simulation-only manifest that replaces the fused TE J35 tabletop "
            "asset with a body plus coaxially rotating coupling nut"
        ),
    )
    parser.add_argument("--preflight-evaluation")
    parser.add_argument("--same-reset-rgbd-config")
    parser.add_argument("--capture-id")
    parser.add_argument(
        "--visual-transport-target",
        help=(
            "versioned ordinary-RGB-D transport target; changes motion-plan "
            "construction only and does not authorize contact"
        ),
    )
    parser.add_argument("--reference-trace")
    parser.add_argument(
        "--robustness-scenario",
        help="named frozen physical perturbation; omitted means nominal",
    )
    parser.add_argument(
        "--preload-increment-rad",
        type=float,
        help="explicit finite all-finger preload increment for a bounded comparison",
    )
    parser.add_argument(
        "--lift-arm-damping-nm-s-rad",
        type=float,
        help="explicit positive arm damping during lift for a bounded comparison",
    )
    parser.add_argument(
        "--finger-preload-scales",
        type=float,
        nargs=3,
        metavar=("F1", "F2", "F3"),
        help="bounded per-finger fractions of the finite preload increment",
    )
    parser.add_argument(
        "--palm-joint-position-rad",
        type=float,
        help="explicit finite palm joint position for an already evaluated grasp",
    )
    parser.add_argument(
        "--grasp-axis-position-m",
        type=float,
        help="explicit connector-axis hand position for an already evaluated grasp",
    )
    parser.add_argument(
        "--hand-yaw-rad",
        type=float,
        help="explicit connector-axis hand yaw for an already evaluated grasp",
    )
    parser.add_argument(
        "--required-closing-joint-effort-nm",
        type=float,
        nargs=3,
        metavar=("F1", "F2", "F3"),
        help="explicit finite pre-lift effort contract in physical finger order",
    )
    parser.add_argument(
        "--closing-order",
        nargs=3,
        choices=("finger_1", "finger_2", "finger_3"),
        metavar=("FIRST", "SECOND", "THIRD"),
        help="one explicit permutation of the three physical fingers",
    )
    parser.add_argument(
        "--contact-coordination-mode",
        choices=("sequential", "parallel_contact_latch"),
        default="sequential",
        help=(
            "sequential baseline or simultaneous low-speed approach with "
            "independent per-finger contact latching"
        ),
    )
    parser.add_argument(
        "--approach-high-seed-arm-positions-rad",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="explicit safe seven-joint IK branch seed for a bounded comparison",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--initialize-at-pregrasp", action="store_true")
    parser.add_argument(
        "--capture-visual-evidence", action="store_true",
        help="save four post-step Isaac viewport frames without feeding image truth to control",
    )
    parser.add_argument(
        "--omit-trace-json", action="store_true",
        help="evaluate in memory but do not duplicate the large trace.json",
    )
    parser.add_argument(
        "--postgrasp-disturbance-panel",
        help="frozen finite postgrasp wrench panel JSON",
    )
    parser.add_argument(
        "--postgrasp-disturbance-condition",
        help="one condition ID from the frozen postgrasp panel",
    )
    parser.add_argument(
        "--nominal-grasp-qualification-evaluation",
        help="prior 50 mm plus 2 s nominal evidence authorizing disturbance",
    )
    arguments = parser.parse_args()
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift") and not arguments.preflight_evaluation:
        parser.error("object contact execution requires --preflight-evaluation")
    if arguments.mode == "isolated-hand" and not arguments.reference_trace:
        parser.error("isolated-hand requires --reference-trace")
    if arguments.mode == "isolated-hand" and arguments.visual_transport_target:
        parser.error("isolated-hand cannot consume a visual transport target")
    if arguments.mode in (SAME_RESET_RGBD_MODE, *VISION_MOTION_MODES):
        if not arguments.same_reset_rgbd_config or not arguments.capture_id:
            parser.error(
                f"{arguments.mode} requires --same-reset-rgbd-config and --capture-id"
            )
        forbidden = {
            "visual_transport_target": arguments.visual_transport_target,
            "preflight_evaluation": arguments.preflight_evaluation,
            "reference_trace": arguments.reference_trace,
            "robustness_scenario": (
                None
                if arguments.mode == VISION_GRASP_SERVO_MODE
                else arguments.robustness_scenario
            ),
            "preload_increment_rad": arguments.preload_increment_rad,
            "lift_arm_damping_nm_s_rad": arguments.lift_arm_damping_nm_s_rad,
            "finger_preload_scales": arguments.finger_preload_scales,
            "palm_joint_position_rad": arguments.palm_joint_position_rad,
            "grasp_axis_position_m": arguments.grasp_axis_position_m,
            "hand_yaw_rad": arguments.hand_yaw_rad,
            "required_closing_joint_effort_nm": arguments.required_closing_joint_effort_nm,
            "closing_order": arguments.closing_order,
            "approach_high_seed_arm_positions_rad": (
                arguments.approach_high_seed_arm_positions_rad
            ),
            "postgrasp_disturbance_panel": arguments.postgrasp_disturbance_panel,
            "postgrasp_disturbance_condition": (
                arguments.postgrasp_disturbance_condition
            ),
            "nominal_grasp_qualification_evaluation": (
                arguments.nominal_grasp_qualification_evaluation
            ),
        }
        used = sorted(name for name, value in forbidden.items() if value is not None)
        if used or arguments.initialize_at_pregrasp or arguments.capture_visual_evidence:
            parser.error(
                f"{arguments.mode} forbids target/trajectory overrides: "
                + ", ".join(used)
            )
        if (
            arguments.mode == VISION_GRASP_SERVO_MODE
            and arguments.robustness_scenario != "friction_lower_0p45"
        ):
            parser.error(
                "vision-grasp-servo requires the existing friction_lower_0p45 scenario"
            )
    if arguments.capture_visual_evidence and arguments.mode == "isolated-hand":
        parser.error("visual evidence capture is only supported in the object scene")
    if arguments.free_split_object_manifest and (
        arguments.object_id != "te_deutsch_d38999_26fj35pn_step"
        or arguments.mode not in (
            "preflight", "first-finger-diagnostic", "grasp-lift"
        )
    ):
        parser.error(
            "--free-split-object-manifest is limited to the TE J35 preflight "
            "and bounded contact/lift modes"
        )
    if arguments.initialize_at_pregrasp and arguments.mode not in (
        "preflight", "first-finger-diagnostic"
    ):
        parser.error("pregrasp initialization is diagnostic-only")
    disturbance_arguments = (
        arguments.postgrasp_disturbance_panel,
        arguments.postgrasp_disturbance_condition,
        arguments.nominal_grasp_qualification_evaluation,
    )
    if any(value is not None for value in disturbance_arguments) and not all(
        value is not None for value in disturbance_arguments
    ):
        parser.error(
            "postgrasp disturbance requires panel, condition, and nominal qualification"
        )
    if any(value is not None for value in disturbance_arguments) and (
        arguments.mode != "grasp-lift" or arguments.initialize_at_pregrasp
    ):
        parser.error("postgrasp disturbance is full grasp-lift only")
    if arguments.closing_order is not None:
        try:
            control.normalized_closing_order(arguments.closing_order)
        except ValueError as error:
            parser.error(str(error))
    for name in ("grasp_axis_position_m", "hand_yaw_rad"):
        value = getattr(arguments, name)
        if value is not None and not np.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    return arguments


def _load_visual_transport_motion_input(
    repository: Path,
    arguments: argparse.Namespace,
    inputs,
    grasp: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Bind one yaw-free RGB-D target to the frozen transport grasp."""

    arguments.visual_transport_target_path = None
    arguments.visual_transport_target_binding = None
    arguments.visual_transport_target_consumption = {
        "provided": False,
        "motion_plan_pose_source": "FROZEN_SCENE_CONTRACT",
        "assembly_key_pose_consumed": False,
        "online_object_or_semantic_truth_used": False,
    }
    if arguments.visual_transport_target is None:
        return None
    visual_friction_only = bool(
        arguments.mode == VISION_GRASP_SERVO_MODE
        and arguments.robustness_scenario_name == "friction_lower_0p45"
        and arguments.robustness_perturbation
        == {"contact_friction_coefficient": 0.45}
    )
    if arguments.robustness_scenario_name != "nominal" and not visual_friction_only:
        raise ValueError(
            "visual transport target allows only nominal pose or the existing "
            "friction-only 0.45 scene"
        )

    target, target_path = load_visual_transport_target(
        arguments.visual_transport_target, repository
    )
    if target.get("object_id") != arguments.object_id:
        raise ValueError("visual transport target object differs from the run")
    relation, _ = load_transport_grasp_relation(
        target["transport_grasp_relation"]["path"], repository
    )
    if relation["source_evidence"]["selected_config_sha256"] != file_sha256(
        inputs.config.path
    ):
        raise ValueError(
            "runner config differs from the config bound by the frozen transport grasp"
        )

    control_plan = grasp["control_plan"]
    numeric_contract = (
        ("object_from_hand_row_major", "object_from_hand_base_row_major"),
        ("approach_direction_object", "approach_direction_object"),
        ("pregrasp_joint_positions_rad", "pregrasp_joint_positions_rad"),
        ("final_joint_positions_rad", "final_joint_positions_rad"),
        (
            "approach_high_seed_arm_positions_rad",
            "approach_high_seed_arm_positions_rad",
        ),
    )
    for control_name, relation_name in numeric_contract:
        current = np.asarray(control_plan[control_name], dtype=np.float64)
        frozen = np.asarray(
            relation[
                "transform" if relation_name in {
                    "object_from_hand_base_row_major",
                    "approach_direction_object",
                } else "hand_contract"
            ][relation_name],
            dtype=np.float64,
        )
        if current.shape != frozen.shape or not np.allclose(
            current, frozen, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"current grasp field {control_name} differs from the frozen "
                "transport relation"
            )
    if list(control_plan["closing_order"]) != list(
        relation["hand_contract"]["closing_order"]
    ):
        raise ValueError("current closing order differs from the frozen transport relation")

    target_relative = str(target_path.relative_to(repository))
    composer_path = (
        repository
        / "src/kcg_connector/kcg_connector/te_transport_grasp_target.py"
    ).resolve()
    arguments.visual_transport_target_path = target_path
    arguments.visual_transport_target_binding = {
        "target_path": target_relative,
        "target_sha256": file_sha256(target_path),
        "provider_result_sha256": target["provider_result"]["sha256"],
        "grasp_relation_sha256": target["transport_grasp_relation"]["sha256"],
        "composer_source_sha256": file_sha256(composer_path),
        "capture_id": target["capture_id"],
        "assembly_key_pose_consumed": False,
        "online_object_or_semantic_truth_used": False,
    }
    return target


def _registered_grasp(
    inputs, object_id: str, dynamic, palm_joint_position_rad: float,
    approach_high_seed_arm_positions_rad: np.ndarray | None,
    grasp_axis_position_m: float | None,
    hand_yaw_rad: float | None,
    closing_order: tuple[str, ...],
) -> Mapping[str, object]:
    generation = generate_axial_pad_intersection_grasp(
        inputs,
        palm_joint_position_rad=palm_joint_position_rad,
        grasp_axis_position_m=grasp_axis_position_m,
        hand_yaw_rad=hand_yaw_rad,
        apply_table_clearance=False,
    )
    control_plan = dict(generation["control_plan"])
    control_plan["closing_order"] = list(closing_order)
    if approach_high_seed_arm_positions_rad is not None:
        control_plan["approach_high_seed_arm_positions_rad"] = (
            approach_high_seed_arm_positions_rad.tolist()
        )
    grasp = {
        "schema_version": "carts_v2_registered_grasp_v1",
        "grasp_id": f"axial_full_pad_first_intersection_v1__{object_id}",
        "object_id": object_id,
        "hardware_authorized": False,
        "construction_method": generation["method"],
        "control_plan": control_plan,
        "generation_evidence": generation["evidence"],
        "closure_control": {
            "method": "SEQUENTIAL_LOW_SPEED_JOINT_EFFORT_CONTACT_THEN_FINITE_PRELOAD",
            "closing_order": list(closing_order),
            "finger_maximum_speed_rad_s": dynamic["finger_maximum_speed_rad_s"],
            "contact_detection_effort_rise_nm": dynamic["contact_effort_rise_nm"],
        },
        "finite_clamp_target": {
            "preload_increment_rad": dynamic["preload_increment_rad"],
            "finger_preload_scales": dynamic["finger_preload_scales"],
            "required_closing_joint_effort_nm": dict(
                zip(
                    CLOSING_JOINT_NAMES,
                    dynamic["required_closing_joint_effort_nm"],
                )
            ),
            "predicted_unit_task_closing_joint_effort_nm": dict(
                zip(
                    CLOSING_JOINT_NAMES,
                    dynamic["required_closing_joint_effort_nm"],
                )
            ),
            "prelift_contact_confirmation_effort_nm": dynamic[
                "contact_effort_rise_nm"
            ],
            "hand_drive_maximum_effort_nm": dynamic[
                "hand_drive_maximum_effort_nm"
            ],
            "measured_effort_abort_nm": dynamic["measured_effort_abort_nm"],
        },
        "lift_trajectory": {
            "direction_world": [0.0, 0.0, 1.0],
            "distance_m": dynamic["lift_command_distance_m"],
            "duration_s": dynamic["lift_duration_s"],
            "arm_damping_nm_s_rad": dynamic["lift_arm_damping_nm_s_rad"],
            "registered_peak_acceleration_m_s2": dynamic[
                "registered_lift_peak_acceleration_m_s2"
            ],
        },
        "hold_control": {
            "method": "HOLD_FINAL_ARM_AND_FINITE_PRELOAD_TARGETS",
            "duration_s": dynamic["hold_duration_s"],
        },
    }
    if not isinstance(grasp, Mapping):
        raise ValueError("object has no directly registered grasp")
    if (
        grasp.get("schema_version") != "carts_v2_registered_grasp_v1"
        or grasp.get("object_id") != object_id
        or grasp.get("hardware_authorized") is not False
        or not isinstance(grasp.get("grasp_id"), str)
        or not isinstance(grasp.get("control_plan"), Mapping)
    ):
        raise ValueError("registered grasp identity or authorization is invalid")
    control_plan = grasp["control_plan"]
    object_from_hand = np.asarray(
        control_plan.get("object_from_hand_row_major"), dtype=np.float64
    )
    pregrasp = np.asarray(
        control_plan.get("pregrasp_joint_positions_rad"), dtype=np.float64
    )
    final = np.asarray(
        control_plan.get("final_joint_positions_rad"), dtype=np.float64
    )
    if (
        object_from_hand.shape != (16,)
        or pregrasp.shape != (4,)
        or final.shape != (4,)
        or not all(np.all(np.isfinite(row)) for row in (object_from_hand, pregrasp, final))
        or not np.allclose(object_from_hand.reshape(4, 4)[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise ValueError("registered grasp control plan is not finite and rigid-shaped")
    clamp = grasp.get("finite_clamp_target")
    lift = grasp.get("lift_trajectory")
    hold = grasp.get("hold_control")
    if not all(isinstance(row, Mapping) for row in (clamp, lift, hold)):
        raise ValueError("registered grasp omits finite clamp, lift, or hold control")
    expected = (
        (clamp["preload_increment_rad"], dynamic["preload_increment_rad"]),
        (clamp["hand_drive_maximum_effort_nm"], dynamic["hand_drive_maximum_effort_nm"]),
        (clamp["measured_effort_abort_nm"], dynamic["measured_effort_abort_nm"]),
        (lift["distance_m"], dynamic["lift_command_distance_m"]),
        (lift["duration_s"], dynamic["lift_duration_s"]),
        (
            lift["arm_damping_nm_s_rad"],
            dynamic["lift_arm_damping_nm_s_rad"],
        ),
        (hold["duration_s"], dynamic["hold_duration_s"]),
    )
    if any(float(left) != float(right) for left, right in expected):
        raise ValueError("registered grasp differs from bounded dynamic control settings")
    observed_required_effort = clamp.get("required_closing_joint_effort_nm")
    if (
        not isinstance(observed_required_effort, Mapping)
        or tuple(observed_required_effort) != CLOSING_JOINT_NAMES
        or not np.allclose(
            [observed_required_effort[name] for name in CLOSING_JOINT_NAMES],
            dynamic["required_closing_joint_effort_nm"],
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("registered grasp required effort contract changed")
    return grasp


def _apply_hand_tilt_about_object_pivot(
    inputs, grasp: Mapping[str, object], perturbation: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply one bounded object-frame tilt to the complete generated hand pose."""

    control_plan = dict(grasp["control_plan"])
    before = np.asarray(
        control_plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    if (
        not np.all(np.isfinite(before))
        or not np.allclose(
            before[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError("generated object-from-hand pose is not one finite transform")

    raw = perturbation.get(HAND_TILT_PERTURBATION_KEY)
    identity = np.eye(4, dtype=np.float64)
    before_rotation = before[:3, :3]
    before_orthogonality_error = float(np.linalg.norm(
        before_rotation.T @ before_rotation - np.eye(3), ord="fro"
    ))
    before_determinant = float(np.linalg.det(before_rotation))
    required_table_clearance = float(
        inputs.config.section("nominal_grasp_generation")[
            "minimum_pregrasp_hand_table_clearance_m"
        ]
    )
    if not np.isfinite(required_table_clearance) or required_table_clearance < 0.0:
        raise ValueError("minimum pregrasp hand-table clearance is invalid")
    if raw is None:
        audit = {
            "schema_version": "carts_v2_hand_tilt_about_object_pivot_audit_v1",
            "applied": False,
            "requested": None,
            "applied_axis_supplier_object": None,
            "applied_angle_rad": 0.0,
            "applied_pivot_supplier_object_m": None,
            "before_object_from_hand_row_major": before.ravel().tolist(),
            "left_multiplier_object_row_major": identity.ravel().tolist(),
            "after_object_from_hand_row_major": before.ravel().tolist(),
            "pivot_fixed_point_error_m": 0.0,
            "left_multiplier_rotation_orthogonality_error": 0.0,
            "left_multiplier_rotation_determinant": 1.0,
            "before_rotation_orthogonality_error": before_orthogonality_error,
            "before_rotation_determinant": before_determinant,
            "after_rotation_orthogonality_error": before_orthogonality_error,
            "after_rotation_determinant": before_determinant,
            "object_scene_transform_modified": False,
            "online_control_used": False,
            "generated_contact_prediction_recomputed_after_tilt": False,
            "pregrasp_hand_table_clearance": {
                "status": "NOT_APPLICABLE_NO_HAND_TILT",
                "required_m": required_table_clearance,
                "observed_m": None,
                "margin_m": None,
                "pass": None,
                "limiting_link": None,
                "comparison_tolerance_m": 1.0e-12,
                "automatic_world_z_shift_applied": False,
                "target_modified_after_check": False,
                "behavior_unchanged": True,
            },
        }
        return dict(grasp), audit

    if not isinstance(raw, Mapping) or set(raw) != HAND_TILT_REQUIRED_KEYS:
        raise ValueError(
            "hand tilt must contain exactly axis, angle, and object-frame pivot"
        )
    axis = np.asarray(raw["axis_supplier_object"], dtype=np.float64)
    pivot = np.asarray(raw["pivot_supplier_object_m"], dtype=np.float64)
    angle = float(raw["angle_rad"])
    axis_norm = float(np.linalg.norm(axis)) if axis.shape == (3,) else np.nan
    if (
        axis.shape != (3,)
        or not np.all(np.isfinite(axis))
        or not np.isfinite(axis_norm)
        or not np.isclose(axis_norm, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("hand tilt axis must be one finite unit object-frame vector")
    if not np.isfinite(angle) or abs(angle) > 0.05:
        raise ValueError("hand tilt angle must be finite and bounded by 0.05 rad")
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
        raise ValueError("hand tilt pivot must be one finite object-frame point")
    object_translation = np.asarray(
        perturbation.get("object_translation_delta_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    object_yaw = float(perturbation.get("object_yaw_delta_rad", 0.0))
    if (
        object_translation.shape != (3,)
        or not np.all(np.isfinite(object_translation))
        or not np.array_equal(object_translation, np.zeros(3))
        or not np.isfinite(object_yaw)
        or object_yaw != 0.0
    ):
        raise ValueError("hand tilt cannot also move the object scene")
    if before_orthogonality_error > 1.0e-10 or not np.isclose(
        before_determinant, 1.0, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("generated object-from-hand rotation is not proper orthogonal")

    applied_axis = axis / axis_norm
    x, y, z = applied_axis
    skew = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64
    )
    rotation = (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    multiplier = np.eye(4, dtype=np.float64)
    multiplier[:3, :3] = rotation
    multiplier[:3, 3] = pivot - rotation @ pivot
    after = multiplier @ before
    pivot_after = rotation @ pivot + multiplier[:3, 3]
    after_rotation = after[:3, :3]
    multiplier_orthogonality_error = float(np.linalg.norm(
        rotation.T @ rotation - np.eye(3), ord="fro"
    ))
    multiplier_determinant = float(np.linalg.det(rotation))
    after_orthogonality_error = float(np.linalg.norm(
        after_rotation.T @ after_rotation - np.eye(3), ord="fro"
    ))
    after_determinant = float(np.linalg.det(after_rotation))
    pivot_error = float(np.linalg.norm(pivot_after - pivot))
    if (
        pivot_error > 1.0e-12
        or multiplier_orthogonality_error > 1.0e-12
        or not np.isclose(multiplier_determinant, 1.0, rtol=0.0, atol=1.0e-12)
        or after_orthogonality_error > 1.0e-10
        or not np.isclose(after_determinant, 1.0, rtol=0.0, atol=1.0e-10)
        or not np.allclose(
            after[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError("applied hand tilt did not preserve one rigid transform")

    control_plan["object_from_hand_row_major"] = after.ravel().tolist()
    updated_grasp = dict(grasp)
    updated_grasp["control_plan"] = control_plan
    pregrasp = np.asarray(
        control_plan["pregrasp_joint_positions_rad"], dtype=np.float64
    )
    if pregrasp.shape != (4,) or not np.all(np.isfinite(pregrasp)):
        raise ValueError("tilted grasp pregrasp joints are not finite")
    pregrasp_transforms = inputs.hand_model.forward_kinematics(pregrasp)
    world_from_object = np.asarray(
        inputs.frozen_world_from_object, dtype=np.float64
    )
    if world_from_object.shape != (4, 4) or not np.all(
        np.isfinite(world_from_object)
    ):
        raise ValueError("frozen world-from-object transform is invalid")
    world_from_hand = world_from_object @ after
    minimum_world_z = np.inf
    limiting_link = ""
    collision_link_count = 0
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        if link_name not in pregrasp_transforms:
            raise ValueError(
                f"pregrasp FK omits collision link {link_name}"
            )
        link_from_local = np.asarray(
            pregrasp_transforms[link_name], dtype=np.float64
        )
        points_local = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
        if (
            link_from_local.shape != (4, 4)
            or not np.all(np.isfinite(link_from_local))
            or not len(points_local)
            or not np.all(np.isfinite(points_local))
        ):
            raise ValueError(
                f"pregrasp collision geometry is invalid for {link_name}"
            )
        points_hand = (
            points_local @ link_from_local[:3, :3].T
            + link_from_local[:3, 3]
        )
        points_world = (
            points_hand @ world_from_hand[:3, :3].T
            + world_from_hand[:3, 3]
        )
        link_minimum_world_z = float(np.min(points_world[:, 2]))
        if link_minimum_world_z < minimum_world_z:
            minimum_world_z = link_minimum_world_z
            limiting_link = str(link_name)
        collision_link_count += 1
    table_top_z = float(inputs.table_top_z_m)
    observed_table_clearance = minimum_world_z - table_top_z
    clearance_margin = observed_table_clearance - required_table_clearance
    clearance_pass = bool(
        observed_table_clearance + 1.0e-12 >= required_table_clearance
    )
    if (
        collision_link_count < 1
        or not limiting_link
        or not np.isfinite(table_top_z)
        or not np.isfinite(observed_table_clearance)
        or not np.isfinite(clearance_margin)
    ):
        raise ValueError("tilted pregrasp hand-table clearance is invalid")
    audit = {
        "schema_version": "carts_v2_hand_tilt_about_object_pivot_audit_v1",
        "applied": True,
        "requested": {
            "axis_supplier_object": axis.tolist(),
            "angle_rad": angle,
            "pivot_supplier_object_m": pivot.tolist(),
        },
        "applied_axis_supplier_object": applied_axis.tolist(),
        "applied_angle_rad": angle,
        "applied_pivot_supplier_object_m": pivot.tolist(),
        "before_object_from_hand_row_major": before.ravel().tolist(),
        "left_multiplier_object_row_major": multiplier.ravel().tolist(),
        "after_object_from_hand_row_major": after.ravel().tolist(),
        "pivot_fixed_point_error_m": pivot_error,
        "left_multiplier_rotation_orthogonality_error": (
            multiplier_orthogonality_error
        ),
        "left_multiplier_rotation_determinant": multiplier_determinant,
        "before_rotation_orthogonality_error": before_orthogonality_error,
        "before_rotation_determinant": before_determinant,
        "after_rotation_orthogonality_error": after_orthogonality_error,
        "after_rotation_determinant": after_determinant,
        "object_scene_transform_modified": False,
        "online_control_used": False,
        "generated_contact_prediction_recomputed_after_tilt": False,
        "pregrasp_hand_table_clearance": {
            "status": "COMPLETE",
            "required_m": required_table_clearance,
            "observed_m": observed_table_clearance,
            "margin_m": clearance_margin,
            "pass": clearance_pass,
            "limiting_link": limiting_link,
            "collision_link_count": collision_link_count,
            "table_top_z_m": table_top_z,
            "minimum_hand_world_z_m": minimum_world_z,
            "comparison_tolerance_m": 1.0e-12,
            "geometry_source": (
                "COMPLETE_HAND_COLLISION_TRIANGLES_AT_PREGRASP_FK"
            ),
            "world_pose_source": "FROZEN_WORLD_FROM_OBJECT_TIMES_TILTED_OBJECT_FROM_HAND",
            "automatic_world_z_shift_applied": False,
            "target_modified_after_check": False,
            "behavior_unchanged": False,
        },
    }
    return updated_grasp, audit


def _require_tilted_pregrasp_hand_table_clearance(
    audit: Mapping[str, object]
) -> None:
    clearance = audit["pregrasp_hand_table_clearance"]
    if audit["applied"] is True and clearance["pass"] is not True:
        raise ValueError(
            "tilted pregrasp hand-table clearance is below the frozen minimum: "
            f"required={clearance['required_m']!r}, "
            f"observed={clearance['observed_m']!r}, "
            f"margin={clearance['margin_m']!r}, "
            f"limiting_link={clearance['limiting_link']!r}"
        )


def _load_postgrasp_disturbance(
    repository: Path, arguments, inputs, grasp, scene_entry
) -> None:
    """Load one preregistered wrench without exposing object truth to control."""

    arguments.postgrasp_disturbance = None
    if arguments.postgrasp_disturbance_panel is None:
        return
    panel_path = Path(arguments.postgrasp_disturbance_panel).resolve()
    qualification_path = Path(
        arguments.nominal_grasp_qualification_evaluation
    ).resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_schema = panel.get("schema_version")
    if (
        panel_schema not in {
            POSTGRASP_DISTURBANCE_SCHEMA,
            POSTGRASP_DISTURBANCE_COM_SCHEMA,
        }
        or panel.get("panel_status") != "FROZEN_BEFORE_FIRST_DYNAMIC_CONDITION"
        or panel.get("hardware_authorized") is not False
    ):
        raise ValueError("postgrasp disturbance panel identity is invalid")
    coordinate = panel.get("coordinate_contract")
    timing = panel.get("timing")
    limits = panel.get("input_limits")
    qualification = panel.get("postrun_qualification")
    grasp_binding = panel.get("nominal_grasp_binding")
    conditions = panel.get("conditions")
    if not all(isinstance(value, Mapping) for value in (
        coordinate, timing, limits, qualification, grasp_binding
    )) or not isinstance(conditions, list) or (
        panel_schema == POSTGRASP_DISTURBANCE_SCHEMA
        and len(conditions) not in (1, 3, 8)
    ) or (
        panel_schema == POSTGRASP_DISTURBANCE_COM_SCHEMA
        and not 1 <= len(conditions) <= 64
    ):
        raise ValueError(
            "postgrasp panel must contain one time control, three independent "
            "conditions, or eight legacy bounded conditions"
        )

    task_from_object = np.asarray(
        coordinate.get("task_from_supplier_object_rotation_row_major"),
        dtype=np.float64,
    )
    task_from_hand = np.asarray(
        coordinate.get("task_from_hand_rotation_row_major"),
        dtype=np.float64,
    )
    world_from_task = (
        np.asarray(
            coordinate.get("frozen_world_from_task_rotation_row_major"),
            dtype=np.float64,
        )
        if panel_schema == POSTGRASP_DISTURBANCE_SCHEMA
        else None
    )
    application_point = np.asarray(
        coordinate.get("application_point_supplier_object_m"),
        dtype=np.float64,
    )
    frame_source = coordinate.get("frozen_world_task_frame_source")
    hand_frozen_frame = bool(
        panel_schema == POSTGRASP_DISTURBANCE_COM_SCHEMA
        and frame_source
        == "CURRENT_RUN_HELD_HAND_POSE_FROZEN_BEFORE_DISTURBANCE"
    )
    rotations = [("task_from_object", task_from_object)]
    if hand_frozen_frame:
        rotations.append(("task_from_hand", task_from_hand))
    if world_from_task is not None:
        rotations.append(("world_from_task", world_from_task))
    for name, rotation in rotations:
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(
                rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-12
            )
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"postgrasp {name} rotation is not proper orthogonal")
    if frame_source == "PRIOR_NOMINAL_FIXED_POSE_RUN_POSTRUN_FINAL_HOLD_POSE":
        def bound_source_path(path_key: str) -> Path:
            raw = Path(str(coordinate.get(path_key, "")))
            if raw.is_absolute():
                raise ValueError("postgrasp source evidence path must be relative")
            resolved = (repository / raw).resolve()
            try:
                resolved.relative_to(repository.resolve())
            except ValueError as error:
                raise ValueError(
                    "postgrasp source evidence escapes repository"
                ) from error
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            return resolved

        source_evaluation = bound_source_path(
            "frozen_world_task_frame_source_evaluation"
        )
        source_trace = bound_source_path(
            "frozen_world_task_frame_source_trace"
        )
        if (
            file_sha256(source_evaluation)
            != coordinate.get(
                "frozen_world_task_frame_source_evaluation_sha256"
            )
            or file_sha256(source_trace)
            != coordinate.get("frozen_world_task_frame_source_trace_sha256")
        ):
            raise ValueError(
                "postgrasp frozen task frame source evidence changed"
            )
        source_sample = coordinate.get(
            "frozen_world_task_frame_source_final_sample"
        )
        if (
            not isinstance(source_sample, Mapping)
            or int(source_sample.get("step", -1)) != 8600
            or float(source_sample.get("simulation_time_s", -1.0)) != 35.8375
            or source_sample.get("phase") != "hold"
        ):
            raise ValueError("postgrasp frozen task frame source sample changed")
        source_orientation = np.asarray(
            source_sample.get("supplier_object_orientation_world_wxyz"),
            dtype=np.float64,
        )
        if (
            source_orientation.shape != (4,)
            or not np.all(np.isfinite(source_orientation))
            or not np.isclose(
                np.linalg.norm(source_orientation),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("postgrasp source object orientation is invalid")
        w, x, y, z = source_orientation / np.linalg.norm(source_orientation)
        world_from_object = np.asarray((
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)),
        ), dtype=np.float64)
        derived_world_from_task = world_from_object @ task_from_object.T
    elif frame_source == "REGISTERED_SCENE_INITIAL_OBJECT_POSE":
        registered_world_from_object = np.asarray(
            scene_entry["frozen_settled_world_from_object_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        derived_world_from_task = (
            registered_world_from_object[:3, :3] @ task_from_object.T
        )
    elif (
        panel_schema == POSTGRASP_DISTURBANCE_COM_SCHEMA
        and frame_source
        in {
            "CURRENT_RUN_POSTGRASP_HOLD_BODY_POSE_FROZEN_BEFORE_DISTURBANCE",
            "CURRENT_RUN_HELD_HAND_POSE_FROZEN_BEFORE_DISTURBANCE",
        }
    ):
        derived_world_from_task = None
    else:
        raise ValueError("postgrasp world task frame source changed")
    if derived_world_from_task is not None and not np.allclose(
        world_from_task, derived_world_from_task, rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("postgrasp frozen world task rotation is not source-derived")
    if panel_schema == POSTGRASP_DISTURBANCE_SCHEMA:
        valid_application_point = coordinate.get("application_point") in (
            "TE_MATING_FACE_CENTER_RIGID_BODY_ORIGIN",
            "TE_BODY_RIGID_BODY_ORIGIN",
        )
        application_valid = bool(
            valid_application_point
            and np.array_equal(application_point, np.zeros(3))
            and coordinate.get("online_object_pose_readback_used") is False
        )
    else:
        if arguments.free_split_object_manifest is None:
            raise ValueError("body-COM disturbance requires the split plug manifest")
        manifest_path = Path(
            arguments.free_split_object_manifest
        ).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_body_com = np.asarray(
            manifest["parts"]["Body"]["mass_properties"]["center_of_mass_m"],
            dtype=np.float64,
        )
        application_valid = bool(
            coordinate.get("application_point") == "TE_BODY_CENTER_OF_MASS"
            and expected_body_com.shape == (3,)
            and np.array_equal(application_point, expected_body_com)
            and coordinate.get(
                "current_body_pose_readback_used_only_for_test_frame_and_com_application"
            ) is True
            and coordinate.get("body_com_source_manifest_sha256")
            == file_sha256(manifest_path)
        )
    if (
        application_point.shape != (3,)
        or not np.all(np.isfinite(application_point))
        or not application_valid
    ):
        raise ValueError("postgrasp application point or truth boundary changed")

    dt = float(arguments.dynamic_settings["physics_dt_s"])
    timing_values = {
        "ramp_up_s": float(timing.get("ramp_up_s", -1.0)),
        "plateau_s": float(timing.get("plateau_s", 0.0)),
        "ramp_down_s": float(timing.get("ramp_down_s", -1.0)),
        "recovery_s": float(timing.get("recovery_s", -1.0)),
    }
    expected_timing = (
        {
            "ramp_up_s": 0.2,
            "plateau_s": 0.0,
            "ramp_down_s": 0.2,
            "recovery_s": 2.0,
        }
        if panel_schema == POSTGRASP_DISTURBANCE_SCHEMA
        else {
            "ramp_up_s": 0.2,
            "plateau_s": 1.0,
            "ramp_down_s": 0.2,
            "recovery_s": 2.0,
        }
    )
    if (
        timing.get("ramp_profile") != "MINIMUM_JERK"
        or int(timing.get("zero_input_baseline_steps", -1)) != 1
        or timing_values != expected_timing
        or any(
            not np.isclose(value / dt, round(value / dt), rtol=0.0, atol=1.0e-12)
            for value in timing_values.values()
        )
    ):
        raise ValueError("postgrasp timing is not the frozen 0.2/0.2/2.0 profile")

    force_cap = float(limits.get("maximum_resultant_force_n", -1.0))
    moment_cap = float(limits.get("maximum_resultant_moment_nm", -1.0))
    simultaneous_allowed = limits.get("force_and_moment_may_be_simultaneous")
    if (
        not 0.0 < force_cap <= 3.0400615
        or not 0.0 < moment_cap <= 0.0540690202
        or not isinstance(simultaneous_allowed, bool)
    ):
        raise ValueError("postgrasp wrench limits changed")
    seen = set()
    validated = []
    for index, raw in enumerate(conditions):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("condition_id"), str):
            raise ValueError("postgrasp condition identity is invalid")
        condition_id = raw["condition_id"]
        if condition_id in seen:
            raise ValueError("postgrasp condition IDs are not unique")
        seen.add(condition_id)
        force_task = np.asarray(raw.get("force_task_n"), dtype=np.float64)
        moment_task = np.asarray(raw.get("moment_task_nm"), dtype=np.float64)
        if (
            force_task.shape != (3,)
            or moment_task.shape != (3,)
            or not np.all(np.isfinite(force_task))
            or not np.all(np.isfinite(moment_task))
            or np.linalg.norm(force_task) > force_cap + 1.0e-12
            or np.linalg.norm(moment_task) > moment_cap + 1.0e-12
        ):
            raise ValueError("postgrasp condition exceeds the frozen wrench norm")
        if (
            simultaneous_allowed is False
            and np.linalg.norm(force_task) > 0.0
            and np.linalg.norm(moment_task) > 0.0
        ):
            raise ValueError(
                "independent postgrasp condition mixes force and moment"
            )
        validated.append({
            "condition_id": condition_id,
            "condition_index": index,
            "description_cn": str(raw.get("description_cn", "")),
            "force_task_n": force_task.tolist(),
            "moment_task_nm": moment_task.tolist(),
            "force_world_n": (
                None
                if world_from_task is None
                else (world_from_task @ force_task).tolist()
            ),
            "moment_world_nm": (
                None
                if world_from_task is None
                else (world_from_task @ moment_task).tolist()
            ),
        })
    selected = [
        row for row in validated
        if row["condition_id"] == arguments.postgrasp_disturbance_condition
    ]
    if len(selected) != 1:
        raise ValueError("postgrasp condition is not in the frozen panel")

    nominal = json.loads(qualification_path.read_text(encoding="utf-8"))
    unauthorized = nominal.get("unauthorized_contact_records", {})
    if arguments.free_split_object_manifest is None:
        nominal_stability_complete = bool(
            nominal.get("te_stability_recording_complete") is True
        )
    else:
        grasp_part_pose = nominal.get("hand_grasp_part_relative_pose", {})
        full_pose = nominal.get("hand_object_full_relative_pose", {})
        split_motion = nominal.get("split_plug_relative_motion", {})
        split_contact = nominal.get("split_plug_grasp_contact_policy", {})
        nominal_stability_complete = bool(
            isinstance(grasp_part_pose, Mapping)
            and grasp_part_pose.get("measurement_complete") is True
            and isinstance(full_pose, Mapping)
            and full_pose.get("measurement_complete") is True
            and isinstance(split_motion, Mapping)
            and split_motion.get("status") == "COMPLETE"
            and isinstance(split_contact, Mapping)
            and split_contact.get(
                "all_three_fingers_contacted_coupling_nut_only"
            ) is True
            and split_contact.get("body_contact_observed") is False
        )
    nominal_physical = bool(
        nominal.get("object_id") == arguments.object_id
        and nominal.get("candidate_id") == grasp["grasp_id"]
        and nominal.get("mode") == "grasp-lift"
        and nominal.get("three_terminal_link_contacts_observed") is True
        and nominal.get("pad_surface_identity_verified") is True
        and float(nominal.get("maximum_lift_m", 0.0)) >= 0.05
        and float(nominal.get("hold_duration_s", 0.0)) >= 2.0
        and nominal.get("table_contact_released_during_hold") is True
        and nominal.get("controller_completed") is True
        and nominal.get("controller_failure_reason") is None
        and nominal_stability_complete
        and isinstance(unauthorized, Mapping)
        and all(int(value) == 0 for value in unauthorized.values())
    )
    binding = nominal.get("evidence_binding", {})
    expected_binding = {
        "config_sha256": file_sha256(inputs.config.path),
        "control_plan_sha256": _json_sha256(grasp["control_plan"]),
        "runtime_resources_sha256": file_sha256(arguments.runtime_resources_path),
        "robot_asset_sha256": file_sha256(arguments.robot_asset_path),
    }
    binding_matches = bool(
        isinstance(binding, Mapping)
        and all(binding.get(key) == value for key, value in expected_binding.items())
        and binding.get("robustness_scenario")
        == arguments.robustness_scenario_name
        and binding.get("effective_finger_preload_scales")
        == arguments.effective_finger_preload_scales
        and np.array_equal(
            np.asarray(
                binding.get("required_closing_joint_effort_nm"),
                dtype=np.float64,
            ),
            np.asarray(
                arguments.required_closing_joint_effort_nm,
                dtype=np.float64,
            ),
        )
        and float(
            nominal.get("lift_arm_damping_switch_audit", {}).get(
                "requested_nm_s_rad", -1.0
            )
        ) == float(arguments.effective_lift_arm_damping_nm_s_rad)
    )
    preflight_binding = arguments.preflight_document.get("evidence_binding", {})
    physical_binding_keys = (
        "config_sha256",
        "control_plan_sha256",
        "runtime_resources_sha256",
        "capacity_audit_sha256",
        "scene_evidence_sha256",
        "environment_scope",
        "robustness_scenario",
        "robustness_perturbation",
        "effective_finger_preload_scales",
        "required_closing_joint_effort_nm",
        "prelift_contact_confirmation_effort_nm",
        "configured_closing_order",
        "effective_closing_order",
        "contact_friction_material_audit",
        "object_mass_audit",
        "center_of_mass_audit",
        "finger_joint_target_audit",
        "hand_tilt_about_object_pivot_audit",
        "object_asset_sha256",
        "robot_asset_sha256",
    )
    preflight_physical_binding_matches = bool(
        isinstance(preflight_binding, Mapping)
        and all(
            preflight_binding.get(key) == binding.get(key)
            for key in physical_binding_keys
        )
    )
    object_asset = (repository / str(scene_entry["asset"])).resolve()
    panel_grasp_binding_matches = bool(
        grasp_binding.get("config_sha256") == file_sha256(inputs.config.path)
        and grasp_binding.get("control_plan_sha256")
        == _json_sha256(grasp["control_plan"])
        and grasp_binding.get("object_asset_sha256") == file_sha256(object_asset)
        and grasp_binding.get("robot_asset_sha256")
        == file_sha256(arguments.robot_asset_path)
        and float(grasp_binding.get("contact_friction_coefficient", -1.0))
        == float(arguments.robustness_perturbation.get(
            "contact_friction_coefficient", -1.0
        ))
        and float(grasp_binding.get("preload_increment_rad", -1.0))
        == float(arguments.effective_preload_increment_rad)
        and grasp_binding.get("finger_preload_scales")
        == arguments.effective_finger_preload_scales
        and float(grasp_binding.get("lift_arm_damping_nm_s_rad", -1.0))
        == float(arguments.effective_lift_arm_damping_nm_s_rad)
        and np.array_equal(
            np.asarray(
                grasp_binding.get("required_closing_joint_effort_nm"),
                dtype=np.float64,
            ),
            np.asarray(
                arguments.required_closing_joint_effort_nm,
                dtype=np.float64,
            ),
        )
    )
    if not (
        nominal_physical
        and binding_matches
        and preflight_physical_binding_matches
        and panel_grasp_binding_matches
    ):
        raise ValueError("matching nominal 50 mm plus 2 s qualification is absent")

    arguments.postgrasp_disturbance = {
        "schema_version": panel_schema,
        "panel_path": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
        "panel_condition_count": len(validated),
        "condition": selected[0],
        "timing": {
            **timing_values,
            "zero_input_baseline_steps": 1,
            "ramp_profile": "MINIMUM_JERK",
        },
        "input_limits": dict(limits),
        "coordinate_contract": dict(coordinate),
        "postrun_qualification": dict(qualification),
        "nominal_qualification_evaluation_path": str(qualification_path),
        "nominal_qualification_evaluation_sha256": file_sha256(
            qualification_path
        ),
        "nominal_physical_qualification_verified": True,
        "nominal_binding_fields_verified": expected_binding,
        "current_preflight_physical_binding_fields_verified": list(
            physical_binding_keys
        ),
        "current_preflight_physical_binding_matches": True,
        "panel_nominal_grasp_binding_matches": True,
        "nominal_registered_grasp_sha256_observed": binding.get(
            "registered_grasp_sha256"
        ),
        "current_registered_grasp_sha256": _json_sha256(grasp),
        "registered_grasp_whole_json_hash_required": False,
        "registered_grasp_whole_json_hash_exclusion_reason": (
            "REDUNDANT_INT_VERSUS_FLOAT_SERIALIZATION; CONFIG_CONTROL_PLAN_"
            "EFFORT_PRELOAD_DAMPING_AND_ASSETS_ARE_BOUND_SEPARATELY"
        ),
        "nominal_source_hashes_used_as_physical_success_claim": False,
        "online_object_pose_readback_used": False,
    }


def _load_plan_inputs(repository: Path, arguments: argparse.Namespace):
    config_path = Path(arguments.config).resolve()
    arguments.runtime_resources_path = Path(arguments.runtime_resources).resolve()
    arguments.runtime_resources_document = load_runtime_resources(
        arguments.runtime_resources_path)
    inputs = load_v2_inputs(repository, config_path=config_path,
                            object_id=arguments.object_id)
    configured_closing_order, _ = control.normalized_closing_order(
        inputs.config.section("closure_prediction")["closing_order"]
    )
    effective_closing_order, _ = control.normalized_closing_order(
        configured_closing_order
        if arguments.closing_order is None
        else arguments.closing_order
    )
    arguments.configured_closing_order = list(configured_closing_order)
    arguments.effective_closing_order = list(effective_closing_order)
    arguments.configured_contact_coordination_mode = "sequential"
    arguments.effective_contact_coordination_mode = str(
        arguments.contact_coordination_mode
    )
    robustness = inputs.config.section("robustness_evaluation")
    scenario_name = arguments.robustness_scenario or "nominal"
    scenarios = robustness.get("scenarios")
    if (
        robustness.get("frozen_before_first_dynamic_run") is not True
        or not isinstance(scenarios, Mapping)
        or scenario_name not in scenarios
        or not isinstance(scenarios[scenario_name], Mapping)
    ):
        raise ValueError("robustness scenario is not in the frozen finite set")
    arguments.robustness_scenario_name = scenario_name
    arguments.robustness_perturbation = dict(scenarios[scenario_name])
    configured_dynamic = inputs.config.section("dynamic")
    dynamic = dict(configured_dynamic)
    dynamic["contact_coordination_mode"] = (
        arguments.effective_contact_coordination_mode
    )
    configured_lift_damping = float(
        configured_dynamic["lift_arm_damping_nm_s_rad"]
    )
    effective_lift_damping = (
        configured_lift_damping
        if arguments.lift_arm_damping_nm_s_rad is None
        else float(arguments.lift_arm_damping_nm_s_rad)
    )
    if not np.isfinite(effective_lift_damping) or effective_lift_damping <= 0.0:
        raise ValueError("lift arm damping must be one positive finite scalar")
    dynamic["lift_arm_damping_nm_s_rad"] = effective_lift_damping
    arguments.configured_lift_arm_damping_nm_s_rad = configured_lift_damping
    arguments.effective_lift_arm_damping_nm_s_rad = effective_lift_damping
    configured_preload = float(configured_dynamic["preload_increment_rad"])
    effective_preload = (
        configured_preload
        if arguments.preload_increment_rad is None
        else float(arguments.preload_increment_rad)
    )
    if not np.isfinite(effective_preload) or effective_preload <= 0.0:
        raise ValueError("preload increment must be one positive finite scalar")
    dynamic["preload_increment_rad"] = effective_preload
    configured_preload_scales = np.ones(3, dtype=np.float64)
    effective_preload_scales = (
        configured_preload_scales
        if arguments.finger_preload_scales is None
        else np.asarray(arguments.finger_preload_scales, dtype=np.float64)
    )
    if (
        effective_preload_scales.shape != (3,)
        or not np.all(np.isfinite(effective_preload_scales))
        or np.any(effective_preload_scales <= 0.0)
        or np.any(effective_preload_scales > 1.0)
    ):
        raise ValueError("finger preload scales must be three finite values in (0, 1]")
    dynamic["finger_preload_scales"] = effective_preload_scales.tolist()
    arguments.dynamic_settings = dynamic
    arguments.configured_preload_increment_rad = configured_preload
    arguments.effective_preload_increment_rad = effective_preload
    arguments.configured_finger_preload_scales = (
        configured_preload_scales.tolist()
    )
    arguments.effective_finger_preload_scales = effective_preload_scales.tolist()
    generation_settings = inputs.config.section("nominal_grasp_generation")
    required_effort_value = generation_settings.get(
        "required_closing_joint_effort_nm"
    )
    if arguments.required_closing_joint_effort_nm is None:
        if (
            not isinstance(required_effort_value, Mapping)
            or tuple(required_effort_value) != CLOSING_JOINT_NAMES
        ):
            raise ValueError(
                "selected grasp requires an explicit three-joint pre-lift effort contract"
            )
        required_effort = np.asarray(
            [required_effort_value[name] for name in CLOSING_JOINT_NAMES],
            dtype=np.float64,
        )
    else:
        required_effort = np.asarray(
            arguments.required_closing_joint_effort_nm, dtype=np.float64
        )
    if (
        required_effort.shape != (3,)
        or not np.all(np.isfinite(required_effort))
        or np.any(required_effort <= 0.0)
        or np.any(required_effort >= float(dynamic["measured_effort_abort_nm"]))
    ):
        raise ValueError(
            "predicted task efforts must be positive and below the abort limit"
        )
    dynamic["required_closing_joint_effort_nm"] = required_effort.tolist()
    arguments.required_closing_joint_effort_nm = required_effort.tolist()
    configured_palm = float(generation_settings["palm_joint_position_rad"])
    effective_palm = (
        configured_palm
        if arguments.palm_joint_position_rad is None
        else float(arguments.palm_joint_position_rad)
    )
    if not np.isfinite(effective_palm):
        raise ValueError("palm joint position must be one finite scalar")
    arguments.configured_palm_joint_position_rad = configured_palm
    arguments.effective_palm_joint_position_rad = effective_palm
    configured_seed_value = generation_settings.get(
        "approach_high_seed_arm_positions_rad"
    )
    configured_seed = (
        None
        if configured_seed_value is None
        else np.asarray(configured_seed_value, dtype=np.float64)
    )
    override_seed = (
        None
        if arguments.approach_high_seed_arm_positions_rad is None
        else np.asarray(
            arguments.approach_high_seed_arm_positions_rad, dtype=np.float64
        )
    )
    effective_seed = configured_seed if override_seed is None else override_seed
    if effective_seed is not None and (
        effective_seed.shape != (7,) or not np.all(np.isfinite(effective_seed))
    ):
        raise ValueError("approach IK branch seed must contain seven finite joints")
    arguments.configured_approach_high_seed_arm_positions_rad = (
        None if configured_seed is None else configured_seed.tolist()
    )
    arguments.effective_approach_high_seed_arm_positions_rad = (
        None if effective_seed is None else effective_seed.tolist()
    )
    registered_robot_asset = (repository / dynamic["robot_asset"]).resolve()
    arguments.robot_asset_path = (
        Path(arguments.robot_asset).expanduser().resolve()
        if arguments.robot_asset
        else registered_robot_asset
    )
    arguments.robot_asset_override_used = bool(arguments.robot_asset)
    grasp = _registered_grasp(
        inputs,
        arguments.object_id,
        dynamic,
        effective_palm,
        effective_seed,
        arguments.grasp_axis_position_m,
        arguments.hand_yaw_rad,
        effective_closing_order,
    )
    grasp, arguments.hand_tilt_about_object_pivot_audit = (
        _apply_hand_tilt_about_object_pivot(
            inputs, grasp, arguments.robustness_perturbation
        )
    )
    _require_tilted_pregrasp_hand_table_clearance(
        arguments.hand_tilt_about_object_pivot_audit
    )
    scene_entry = dynamic["object_scenes"].get(arguments.object_id)
    if not isinstance(scene_entry, dict):
        raise ValueError("object has no registered free tabletop dynamic scene")
    scene_entry = dict(scene_entry)
    if arguments.free_split_object_manifest:
        manifest_path = Path(
            arguments.free_split_object_manifest
        ).expanduser().resolve()
        try:
            manifest_relative = manifest_path.relative_to(repository)
        except ValueError as error:
            raise ValueError("split object manifest escapes the repository") from error
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_manifest = {
            "schema_version": "kcg_te_j35_free_split_tabletop_asset_v1",
            "product_id": "D38999/26FJ35PN",
            "hardware_authorized": False,
            "legal_grasp_contact_part": "CouplingNut",
        }
        if any(
            manifest.get(key) != value
            for key, value in required_manifest.items()
        ):
            raise ValueError("split object manifest identity or safety scope differs")
        asset_path = (repository / str(manifest["asset"])).resolve()
        if (
            not asset_path.is_file()
            or file_sha256(asset_path) != manifest.get("asset_sha256")
        ):
            raise ValueError("split object asset differs from its manifest")
        scene_entry.update(
            {
                "scene_kind": FREE_SPLIT_SCENE_KIND,
                "asset": str(manifest["asset"]),
                "manifest": str(manifest_relative),
                "reference_prim_path": str(manifest["reference_prim_path"]),
                "body_relative_prim_path": str(
                    manifest["body_relative_prim_path"]
                ),
                "coupling_nut_relative_prim_path": str(
                    manifest["coupling_nut_relative_prim_path"]
                ),
                "joint_relative_prim_path": str(
                    manifest["joint_relative_prim_path"]
                ),
                "split_joint_contract": dict(manifest["joint"]),
                "component_bottom_offsets_m": list(
                    manifest["component_bottom_offsets_m"]
                ),
                "legal_grasp_contact_part": str(
                    manifest["legal_grasp_contact_part"]
                ),
            }
        )
    if arguments.mode in (SAME_RESET_RGBD_MODE, *VISION_MOTION_MODES):
        _load_same_reset_rgbd_config(
            repository, arguments, inputs, scene_entry
        )
        arguments.visual_transport_target_path = None
        arguments.visual_transport_target_binding = None
        arguments.visual_transport_target_consumption = {
            "provided": False,
            "motion_plan_pose_source": None,
            "motion_plan_constructed": False,
            "assembly_key_pose_consumed": False,
            "online_object_or_semantic_truth_used": False,
        }
        arguments.postgrasp_disturbance = None
        arguments.finger_joint_target_audit = {
            "applied": False,
            "reason": (
                "FRESH_RGBD_MUST_BE_SEALED_BEFORE_MOTION_PLAN_CONSTRUCTION"
                if arguments.mode in VISION_MOTION_MODES
                else "SAME_RESET_RGBD_OBSERVE_HAS_NO_MOTION_PLAN_OR_TARGETS"
            ),
        }
        return inputs, grasp, scene_entry, None
    visual_transport_target = _load_visual_transport_motion_input(
        repository, arguments, inputs, grasp
    )
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift"):
        arguments.preflight_evaluation_path = Path(
            arguments.preflight_evaluation).resolve()
        preflight = json.loads(arguments.preflight_evaluation_path.read_text(
            encoding="utf-8"))
        arguments.preflight_document = preflight
        expected = (arguments.object_id, grasp["grasp_id"])
        observed = (preflight.get("object_id"), preflight.get("candidate_id"))
        if (
            observed != expected
            or not preflight_is_accepted(preflight)
            or bool(preflight.get("initialized_at_pregrasp"))
            != bool(arguments.initialize_at_pregrasp)
        ):
            raise ValueError("matching independent preflight did not pass")
        preflight_visual_binding = preflight.get("evidence_binding", {}).get(
            "visual_transport_target"
        )
        if preflight_visual_binding != arguments.visual_transport_target_binding:
            raise ValueError(
                "preflight visual transport target does not match this run"
            )
    _load_postgrasp_disturbance(
        repository, arguments, inputs, grasp, scene_entry
    )
    if arguments.mode == "isolated-hand":
        arguments.reference_document = json.loads(
            Path(arguments.reference_trace).read_text(encoding="utf-8"))
        motion_plan = arguments.reference_document["motion_plan"]
    else:
        world_from_object = np.asarray(
            scene_entry["frozen_settled_world_from_object_row_major"]
            if visual_transport_target is None
            else visual_transport_target[
                "world_from_transport_object_row_major"
            ],
            dtype=np.float64,
        ).reshape(4, 4)
        motion_plan = control.build_joint_motion_plan(
            repository, inputs, grasp["control_plan"], world_from_object,
            include_lift=arguments.mode == "grasp-lift",
        )
        if visual_transport_target is not None:
            expected_target = np.asarray(
                visual_transport_target[
                    "world_from_hand_base_target_row_major"
                ],
                dtype=np.float64,
            )
            planned_target = np.asarray(
                motion_plan["world_from_hand_base_target"], dtype=np.float64
            )
            expected_approach = np.asarray(
                visual_transport_target["approach_direction_world"],
                dtype=np.float64,
            )
            planned_approach = np.asarray(
                motion_plan["approach_direction_world"], dtype=np.float64
            )
            if (
                expected_target.shape != (16,)
                or planned_target.shape != (16,)
                or not np.allclose(
                    planned_target, expected_target, rtol=0.0, atol=1.0e-12
                )
                or expected_approach.shape != (3,)
                or planned_approach.shape != (3,)
                or not np.allclose(
                    planned_approach, expected_approach, rtol=0.0, atol=1.0e-12
                )
            ):
                raise ValueError(
                    "motion plan did not consume the bound visual transport target"
                )
            arguments.visual_transport_target_consumption = {
                "provided": True,
                "motion_plan_pose_source": (
                    "ORDINARY_RGBD_TRANSPORT_GRASP_POSE_POSITION_AND_AXIS"
                ),
                "target_binding": arguments.visual_transport_target_binding,
                "world_from_transport_object_row_major": (
                    visual_transport_target[
                        "world_from_transport_object_row_major"
                    ]
                ),
                "planned_world_from_hand_base_target_row_major": (
                    planned_target.tolist()
                ),
                "planned_approach_direction_world": planned_approach.tolist(),
                "target_matrix_matches_frozen_composition": True,
                "assembly_key_pose_consumed": False,
                "scene_contract_pose_used_for_motion_plan": False,
                "online_object_or_semantic_truth_used": False,
                "controller_execution_observed": False,
            }
    joint_offset_value = arguments.robustness_perturbation.get(
        "finger_joint_target_offset_rad"
    )
    joint_offsets = {} if joint_offset_value is None else joint_offset_value
    if not isinstance(joint_offsets, Mapping) or set(joint_offsets) - {"finger_3"}:
        raise ValueError("only the frozen finger-3 joint target offset is supported")
    finger_3_offset = float(joint_offsets.get("finger_3", 0.0))
    finger_3_bounds = np.asarray(
        robustness["bounds"]["finger_3_joint_target_offset_rad"], dtype=np.float64
    )
    if (
        finger_3_bounds.shape != (2,)
        or not np.isfinite(finger_3_offset)
        or finger_3_offset < float(finger_3_bounds[0])
        or finger_3_offset > float(finger_3_bounds[1])
    ):
        raise ValueError("finger-3 joint target offset is outside the frozen bound")
    before_pregrasp = np.asarray(
        motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
    )
    before_final = np.asarray(
        motion_plan["final_hand_positions_rad"], dtype=np.float64
    )
    after_pregrasp = before_pregrasp.copy()
    after_final = before_final.copy()
    after_pregrasp[3] += finger_3_offset
    after_final[3] += finger_3_offset
    motion_plan = dict(motion_plan)
    motion_plan["pregrasp_hand_positions_rad"] = tuple(after_pregrasp)
    motion_plan["final_hand_positions_rad"] = tuple(after_final)
    arguments.finger_joint_target_audit = {
        "applied": joint_offset_value is not None,
        "joint_name": "f3j2",
        "requested_offset_rad": finger_3_offset,
        "before_pregrasp_rad": float(before_pregrasp[3]),
        "observed_pregrasp_target_rad": float(after_pregrasp[3]),
        "before_final_rad": float(before_final[3]),
        "observed_final_target_rad": float(after_final[3]),
    }
    return inputs, grasp, scene_entry, motion_plan


def prepare_dynamic_scene(
    repository: Path, stage, entry, add_reference_to_stage, perturbation
) -> dict[str, object]:
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_environment,
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )

    allowed_keys = {
        "object_translation_delta_m",
        "object_yaw_delta_rad",
        "contact_friction_coefficient",
        "object_mass_scale",
        "center_of_mass_delta_object_m",
        "finger_joint_target_offset_rad",
        HAND_TILT_PERTURBATION_KEY,
    }
    unsupported = set(perturbation) - allowed_keys
    if unsupported:
        raise ValueError(
            "selected robustness scenario is not implemented by the current "
            f"minimal runner: {sorted(unsupported)}"
        )
    translation_delta = np.asarray(
        perturbation.get("object_translation_delta_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if translation_delta.shape != (3,) or not np.all(np.isfinite(translation_delta)):
        raise ValueError("object translation perturbation must be one finite vector")
    yaw_delta_rad = float(perturbation.get("object_yaw_delta_rad", 0.0))
    if not np.isfinite(yaw_delta_rad):
        raise ValueError("object yaw perturbation must be one finite scalar")
    yaw_delta_degrees = float(np.degrees(yaw_delta_rad))
    friction_value = perturbation.get("contact_friction_coefficient")
    contact_friction_coefficient = (
        None if friction_value is None else float(friction_value)
    )
    if (
        contact_friction_coefficient is not None
        and (
            not np.isfinite(contact_friction_coefficient)
            or contact_friction_coefficient < 0.0
        )
    ):
        raise ValueError("contact friction perturbation must be one nonnegative scalar")
    mass_scale_value = perturbation.get("object_mass_scale")
    object_mass_scale = (
        None if mass_scale_value is None else float(mass_scale_value)
    )
    if (
        object_mass_scale is not None
        and (not np.isfinite(object_mass_scale) or object_mass_scale <= 0.0)
    ):
        raise ValueError("object mass perturbation must be one positive finite scale")
    com_delta_value = perturbation.get("center_of_mass_delta_object_m")
    center_of_mass_delta_object_m = (
        None
        if com_delta_value is None
        else np.asarray(com_delta_value, dtype=np.float64)
    )
    if (
        center_of_mass_delta_object_m is not None
        and (
            center_of_mass_delta_object_m.shape != (3,)
            or not np.all(np.isfinite(center_of_mass_delta_object_m))
        )
    ):
        raise ValueError("center-of-mass perturbation must be one finite vector")

    kind = str(entry["scene_kind"])
    if kind == "D38999_PAIR_TABLETOP":
        scene_path = (repository / entry["scene_config"]).resolve()
        scene = load_d38999_tabletop_scene(scene_path)
        asset = verify_d38999_tabletop_asset(scene, repository)
        author_d38999_tabletop_scene(
            stage, scene, asset, add_reference_to_stage=add_reference_to_stage,
            Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
            UsdShade=UsdShade, physics_utils=physicsUtils)
        loose_root = stage.GetPrimAtPath(scene.asset.loose_plug_prim_path)
        loose_ops = UsdGeom.Xformable(loose_root).GetOrderedXformOps()
        translate_ops = [
            operation for operation in loose_ops
            if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        rotate_ops = [
            operation for operation in loose_ops
            if operation.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
        ]
        if len(translate_ops) != 1 or len(rotate_ops) != 1:
            raise RuntimeError(
                "loose plug does not have one editable translation and rotation"
            )
        initial_translation = np.asarray(translate_ops[0].Get(), dtype=np.float64)
        initial_rotation = np.asarray(rotate_ops[0].Get(), dtype=np.float64)
        translate_ops[0].Set(Gf.Vec3d(*(initial_translation + translation_delta)))
        rotate_ops[0].Set(
            Gf.Vec3f(
                float(initial_rotation[0]),
                float(initial_rotation[1]),
                float(initial_rotation[2] + yaw_delta_degrees),
            )
        )
        return {
            "object_asset": asset,
            "part_prim_paths": (scene.asset.body_prim_path, scene.asset.nut_prim_path),
            "part_bottom_offsets_m": (
                scene.loose_endpoint.body_bottom_offset_m,
                scene.loose_endpoint.nut_bottom_offset_m,
            ),
            "roots": {
                "object": scene.asset.loose_plug_prim_path,
                "table": scene.table.prim_path,
                "fixture": scene.fixed_endpoint.fixture_prim_path,
            },
            "table_top_z_m": scene.table.top_z_m,
            "gravity_m_s2": scene.physics.gravity_m_s2,
            "render": scene.render,
            "evidence_paths": (scene_path,),
            "environment_scope": "FULL_TABLE_FIXTURE_AND_FIXED_RECEPTACLE",
            "applied_initial_object_translation_delta_m": translation_delta.tolist(),
            "applied_initial_object_yaw_delta_rad": yaw_delta_rad,
            "requested_contact_friction_coefficient": contact_friction_coefficient,
            "requested_object_mass_scale": object_mass_scale,
            "requested_center_of_mass_delta_object_m": (
                None
                if center_of_mass_delta_object_m is None
                else center_of_mass_delta_object_m.tolist()
            ),
        }
    if kind not in (
        "FREE_SINGLE_RIGID_ON_SHARED_FINITE_TABLE", FREE_SPLIT_SCENE_KIND
    ):
        raise ValueError(f"unsupported dynamic scene kind: {kind}")

    environment_path = (repository / entry["environment_scene_config"]).resolve()
    environment = load_d38999_tabletop_scene(environment_path)
    manifest_path = (repository / entry["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = (repository / entry["asset"]).resolve()
    expected_schema = (
        "kcg_te_j35_free_split_tabletop_asset_v1"
        if kind == FREE_SPLIT_SCENE_KIND
        else "kcg_te_j35_free_tabletop_asset_v1"
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("hardware_authorized") is not False
        or manifest.get("product_id") != "D38999/26FJ35PN"
        or manifest.get("asset_sha256") != file_sha256(asset)
    ):
        raise ValueError("free rigid asset differs from its registered manifest")
    root = str(entry["reference_prim_path"])
    author_d38999_tabletop_environment(
        stage, environment, Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics, UsdShade=UsdShade, physics_utils=physicsUtils)
    add_reference_to_stage(str(asset), root)
    object_prim = stage.GetPrimAtPath(root)
    if not object_prim.IsValid():
        raise RuntimeError("free object root is missing")
    matrix = np.asarray(entry["frozen_settled_world_from_object_row_major"],
                        dtype=np.float64).reshape(4, 4)
    cosine = float(np.cos(yaw_delta_rad))
    sine = float(np.sin(yaw_delta_rad))
    yaw_rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    matrix[:3, :3] = yaw_rotation @ matrix[:3, :3]
    matrix[:3, 3] += translation_delta
    xformable = UsdGeom.Xformable(object_prim)
    if xformable.GetOrderedXformOps():
        raise RuntimeError("free object root already has a transform stack")
    xformable.AddTransformOp().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    if kind == FREE_SPLIT_SCENE_KIND:
        body_path = root + "/" + str(entry["body_relative_prim_path"]).strip("/")
        nut_path = root + "/" + str(
            entry["coupling_nut_relative_prim_path"]
        ).strip("/")
        joint_path = root + "/" + str(entry["joint_relative_prim_path"]).strip("/")
        body_prim = stage.GetPrimAtPath(body_path)
        nut_prim = stage.GetPrimAtPath(nut_path)
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not all(
            prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            for prim in (body_prim, nut_prim)
        ):
            raise RuntimeError("split plug body or coupling nut rigid body is missing")
        if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError("split plug coaxial revolute joint is missing")
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        if (
            joint.GetAxisAttr().Get() != UsdPhysics.Tokens.z
            or tuple(map(str, joint.GetBody0Rel().GetTargets())) != (body_path,)
            or tuple(map(str, joint.GetBody1Rel().GetTargets())) != (nut_path,)
            or joint.GetLowerLimitAttr().HasAuthoredValueOpinion()
            or joint.GetUpperLimitAttr().HasAuthoredValueOpinion()
        ):
            raise RuntimeError("split plug joint is not unlimited coaxial rotation")
        joint_contract = entry.get("split_joint_contract", {})
        resistance_contract = joint_contract.get("rotational_resistance", {})
        drive_expected = bool(joint_contract.get("drive_authored", False))
        drive_present = joint_prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        if drive_present != drive_expected:
            raise RuntimeError("split plug resistance drive presence differs")
        if drive_present:
            drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
            readback = {
                "model": resistance_contract.get("model"),
                "assumed_resisting_torque_nm": float(
                    resistance_contract["assumed_resisting_torque_nm"]
                ),
                "drive_type": str(drive.GetTypeAttr().Get()),
                "target_velocity_deg_s": float(
                    drive.GetTargetVelocityAttr().Get()
                ),
                "stiffness_nm_per_deg": float(drive.GetStiffnessAttr().Get()),
                "damping_nm_s_per_deg": float(drive.GetDampingAttr().Get()),
                "maximum_drive_torque_nm": float(drive.GetMaxForceAttr().Get()),
                "angle_preference_or_limit_authored": False,
                "physical_status": resistance_contract.get("physical_status"),
                "asset_readback_matches_manifest": True,
            }
            numeric_names = (
                "target_velocity_deg_s",
                "stiffness_nm_per_deg",
                "damping_nm_s_per_deg",
                "maximum_drive_torque_nm",
            )
            if (
                readback["drive_type"] != resistance_contract.get("drive_type")
                or any(
                    not math.isclose(
                        readback[name],
                        float(resistance_contract[name]),
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-9,
                    )
                    for name in numeric_names
                )
                or joint_prim.GetAttribute(
                    "drive:angular:physics:targetPosition"
                ).HasAuthoredValueOpinion()
            ):
                raise RuntimeError("split plug resistance drive differs from manifest")
        else:
            readback = {
                "model": "UNMODELED",
                "assumed_resisting_torque_nm": 0.0,
                "asset_readback_matches_manifest": True,
            }
        part_paths = (body_path, nut_path)
        legal_contact_paths = (nut_path,)
    else:
        if not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("free object rigid body is missing")
        part_paths = (root,)
        legal_contact_paths = (root,)
    return {
        "object_asset": asset,
        "part_prim_paths": part_paths,
        "legal_grasp_contact_paths": legal_contact_paths,
        "part_bottom_offsets_m": tuple(entry["component_bottom_offsets_m"]),
        "roots": {
            "object": root,
            "table": environment.table.prim_path,
            "fixture": environment.fixed_endpoint.fixture_prim_path,
        },
        "table_top_z_m": environment.table.top_z_m,
        "gravity_m_s2": environment.physics.gravity_m_s2,
        "render": environment.render,
        "evidence_paths": (environment_path, manifest_path),
        "environment_scope": "SHARED_FINITE_TABLE_AND_FIXTURE_WITHOUT_FIXED_RECEPTACLE",
        "relative_motion_model": (
            "BODY_PLUS_UNLIMITED_COAXIAL_COUPLING_NUT_REVOLUTE"
            if kind == FREE_SPLIT_SCENE_KIND
            else "FUSED_SINGLE_RIGID_BODY"
        ),
        "joint_rotational_resistance": (
            readback if kind == FREE_SPLIT_SCENE_KIND else None
        ),
        "applied_initial_object_translation_delta_m": translation_delta.tolist(),
        "applied_initial_object_yaw_delta_rad": yaw_delta_rad,
        "requested_contact_friction_coefficient": contact_friction_coefficient,
        "requested_object_mass_scale": object_mass_scale,
        "requested_center_of_mass_delta_object_m": (
            None
            if center_of_mass_delta_object_m is None
            else center_of_mass_delta_object_m.tolist()
        ),
    }


def _apply_contact_friction_perturbation(
    stage, scene, Usd, UsdPhysics, UsdShade, PhysxSchema
) -> None:
    material_prim = stage.GetPrimAtPath(FINGERTIP_PHYSICS_MATERIAL_PATH)
    if (
        not material_prim.IsValid()
        or not material_prim.HasAPI(UsdPhysics.MaterialAPI)
    ):
        raise RuntimeError("full-pad fingertip physics material is missing")
    material = UsdPhysics.MaterialAPI(material_prim)
    static_attribute = material.GetStaticFrictionAttr()
    dynamic_attribute = material.GetDynamicFrictionAttr()
    before_static = float(static_attribute.Get())
    before_dynamic = float(dynamic_attribute.Get())
    fingertip_physx = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    fingertip_combine_attribute = fingertip_physx.GetFrictionCombineModeAttr()
    before_fingertip_combine = fingertip_combine_attribute.Get()
    requested = scene["requested_contact_friction_coefficient"]

    object_collision_prims = {}
    for root_path in scene["part_prim_paths"]:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"connector rigid part is missing: {root_path}")
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                object_collision_prims[str(prim.GetPath())] = prim
    if not object_collision_prims:
        raise RuntimeError("connector has no collision surfaces for friction binding")

    connector_material_path = None
    observed_connector_static = None
    observed_connector_dynamic = None
    observed_connector_combine = None
    if requested is not None:
        static_attribute.Set(float(requested))
        dynamic_attribute.Set(float(requested))
        fingertip_combine_attribute.Set(PhysxSchema.Tokens.max)

        connector_material = UsdShade.Material.Define(
            stage, CONNECTOR_CONTACT_MATERIAL_PATH
        )
        connector_material_prim = connector_material.GetPrim()
        connector_api = UsdPhysics.MaterialAPI.Apply(connector_material_prim)
        connector_api.CreateStaticFrictionAttr(float(requested))
        connector_api.CreateDynamicFrictionAttr(float(requested))
        connector_api.CreateRestitutionAttr(0.0)
        connector_physx = PhysxSchema.PhysxMaterialAPI.Apply(
            connector_material_prim
        )
        connector_physx.CreateFrictionCombineModeAttr().Set(
            PhysxSchema.Tokens.max
        )
        for root_path in scene["part_prim_paths"]:
            root_prim = stage.GetPrimAtPath(root_path)
            UsdShade.MaterialBindingAPI.Apply(root_prim).Bind(
                connector_material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
        connector_material_path = str(connector_material.GetPath())
        observed_connector_static = float(
            connector_api.GetStaticFrictionAttr().Get()
        )
        observed_connector_dynamic = float(
            connector_api.GetDynamicFrictionAttr().Get()
        )
        observed_connector_combine = str(
            connector_physx.GetFrictionCombineModeAttr().Get()
        )
    observed_static = float(static_attribute.Get())
    observed_dynamic = float(dynamic_attribute.Get())
    observed_fingertip_combine = str(fingertip_combine_attribute.Get())

    resolved_object_bindings = {}
    for collider_path, prim in object_collision_prims.items():
        bound_material, _ = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial("physics")
        resolved_object_bindings[collider_path] = (
            str(bound_material.GetPath()) if bound_material else None
        )

    table_root_path = str(scene["roots"]["table"])
    table_root_prim = stage.GetPrimAtPath(table_root_path)
    if not table_root_prim.IsValid():
        raise RuntimeError("table root is missing during material readback")
    table_collision_prims = {
        str(prim.GetPath()): prim
        for prim in Usd.PrimRange(
            table_root_prim, Usd.TraverseInstanceProxies()
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    if not table_collision_prims:
        raise RuntimeError("table has no collision surfaces for material readback")
    resolved_table_bindings = {}
    for collider_path, prim in table_collision_prims.items():
        bound_material, _ = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial("physics")
        resolved_table_bindings[collider_path] = (
            str(bound_material.GetPath()) if bound_material else None
        )
    if any(path is None for path in resolved_table_bindings.values()):
        raise RuntimeError("table collision surface has no bound physics material")
    table_material_records = []
    for material_path in sorted(set(resolved_table_bindings.values())):
        table_material_prim = stage.GetPrimAtPath(material_path)
        if (
            not table_material_prim.IsValid()
            or not table_material_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            raise RuntimeError("bound table physics material is invalid")
        table_api = UsdPhysics.MaterialAPI(table_material_prim)
        table_physx = PhysxSchema.PhysxMaterialAPI(table_material_prim)
        table_static = float(table_api.GetStaticFrictionAttr().Get())
        table_dynamic = float(table_api.GetDynamicFrictionAttr().Get())
        table_combine_value = table_physx.GetFrictionCombineModeAttr().Get()
        if not (
            np.isfinite(table_static)
            and np.isfinite(table_dynamic)
            and table_static >= 0.0
            and table_dynamic >= 0.0
        ):
            raise RuntimeError("table material friction readback is invalid")
        table_material_records.append({
            "material_prim_path": material_path,
            "observed_static_friction": table_static,
            "observed_dynamic_friction": table_dynamic,
            "observed_friction_combine_mode": (
                None if table_combine_value is None
                else str(table_combine_value)
            ),
        })

    effective_static = None
    effective_dynamic = None
    connector_table_pairs = []
    if requested is not None:
        effective_static = max(observed_static, observed_connector_static)
        effective_dynamic = max(observed_dynamic, observed_connector_dynamic)
        values_match = np.allclose(
            (
                observed_static,
                observed_dynamic,
                observed_connector_static,
                observed_connector_dynamic,
                effective_static,
                effective_dynamic,
            ),
            (requested,) * 6,
            rtol=0.0,
            atol=1.0e-7,
        )
        combine_modes_match = (
            observed_fingertip_combine == str(PhysxSchema.Tokens.max)
            and observed_connector_combine == str(PhysxSchema.Tokens.max)
        )
        bindings_match = all(
            path == connector_material_path
            for path in resolved_object_bindings.values()
        )
        if not (values_match and combine_modes_match and bindings_match):
            raise RuntimeError(
                "effective fingertip-connector friction perturbation did not read back"
            )
        connector_table_pairs = [
            {
                "connector_material_prim_path": connector_material_path,
                "table_material_prim_path": row["material_prim_path"],
                "effective_combine_rule": "max",
                "effective_static_friction": max(
                    observed_connector_static,
                    row["observed_static_friction"],
                ),
                "effective_dynamic_friction": max(
                    observed_connector_dynamic,
                    row["observed_dynamic_friction"],
                ),
                "formula": "max(connector_side, table_side)",
                "connector_max_mode_dominates_pair": True,
            }
            for row in table_material_records
        ]
    scene["contact_friction_material_audit"] = {
        "applied": requested is not None,
        "requested_coefficient": requested,
        "fingertip_side": {
            "material_prim_path": FINGERTIP_PHYSICS_MATERIAL_PATH,
            "before_static_friction": before_static,
            "before_dynamic_friction": before_dynamic,
            "before_friction_combine_mode": str(before_fingertip_combine),
            "observed_static_friction": observed_static,
            "observed_dynamic_friction": observed_dynamic,
            "observed_friction_combine_mode": observed_fingertip_combine,
            "scope": "THREE_FULL_NAILFREE_FINGERTIP_PADS_SHARED_PHYSICS_MATERIAL",
        },
        "connector_side": {
            "material_prim_path": connector_material_path,
            "part_binding_roots": list(scene["part_prim_paths"]),
            "collision_surface_count": len(object_collision_prims),
            "observed_static_friction": observed_connector_static,
            "observed_dynamic_friction": observed_connector_dynamic,
            "observed_friction_combine_mode": observed_connector_combine,
            "resolved_bound_material_paths": sorted(
                set(resolved_object_bindings.values()),
                key=lambda value: "" if value is None else value,
            ),
        },
        "contact_pair": {
            "combine_rule": "max",
            "effective_static_friction": effective_static,
            "effective_dynamic_friction": effective_dynamic,
            "formula": "max(fingertip_side, connector_side)",
        },
        "table_side": {
            "root_prim_path": table_root_path,
            "collision_surface_count": len(table_collision_prims),
            "resolved_bound_material_paths": sorted(
                set(resolved_table_bindings.values())
            ),
            "materials": table_material_records,
        },
        "connector_table_contact_pairs": connector_table_pairs,
    }


def _apply_object_mass_perturbation(stage, scene, Gf, UsdPhysics) -> None:
    requested = scene["requested_object_mass_scale"]
    scale = 1.0 if requested is None else float(requested)
    records = []
    for path in scene["part_prim_paths"]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"object rigid part lacks explicit mass properties: {path}")
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attribute = mass_api.GetMassAttr()
        inertia_attribute = mass_api.GetDiagonalInertiaAttr()
        before_mass = mass_attribute.Get()
        before_inertia_value = inertia_attribute.Get()
        if before_mass is None or before_inertia_value is None:
            raise RuntimeError(f"object rigid part has incomplete mass properties: {path}")
        before_mass = float(before_mass)
        before_inertia = np.asarray(before_inertia_value, dtype=np.float64)
        if (
            not np.isfinite(before_mass)
            or before_mass <= 0.0
            or before_inertia.shape != (3,)
            or not np.all(np.isfinite(before_inertia))
            or np.any(before_inertia <= 0.0)
        ):
            raise RuntimeError(f"object rigid part has invalid mass properties: {path}")
        if requested is not None:
            mass_attribute.Set(before_mass * scale)
            inertia_attribute.Set(Gf.Vec3f(*(before_inertia * scale)))
        observed_mass = float(mass_attribute.Get())
        observed_inertia = np.asarray(
            inertia_attribute.Get(), dtype=np.float64
        )
        if not np.allclose(
            np.concatenate(([observed_mass], observed_inertia)),
            np.concatenate(([before_mass * scale], before_inertia * scale)),
            rtol=1.0e-6,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"object mass perturbation did not read back: {path}")
        records.append({
            "rigid_prim_path": str(path),
            "before_mass_kg": before_mass,
            "observed_mass_kg": observed_mass,
            "before_diagonal_inertia_kg_m2": before_inertia.tolist(),
            "observed_diagonal_inertia_kg_m2": observed_inertia.tolist(),
        })
    scene["object_mass_audit"] = {
        "applied": requested is not None,
        "requested_scale": requested,
        "effective_scale": scale,
        "parts": records,
    }


def _apply_center_of_mass_perturbation(stage, scene, Gf, UsdGeom, UsdPhysics) -> None:
    requested = scene["requested_center_of_mass_delta_object_m"]
    delta = np.zeros(3, dtype=np.float64) if requested is None else np.asarray(
        requested, dtype=np.float64
    )
    records = []
    object_root = str(scene["roots"]["object"])
    for path in scene["part_prim_paths"]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"object rigid part lacks explicit mass properties: {path}")
        if str(path) != object_root and UsdGeom.Xformable(prim).GetOrderedXformOps():
            raise RuntimeError(
                f"object part axes differ from the object frame: {path}"
            )
        center_attribute = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr()
        before_value = center_attribute.Get()
        if before_value is None:
            raise RuntimeError(f"object rigid part lacks an explicit center of mass: {path}")
        before = np.asarray(before_value, dtype=np.float64)
        if before.shape != (3,) or not np.all(np.isfinite(before)):
            raise RuntimeError(f"object rigid part has an invalid center of mass: {path}")
        if requested is not None:
            center_attribute.Set(Gf.Vec3f(*(before + delta)))
        observed = np.asarray(center_attribute.Get(), dtype=np.float64)
        if not np.allclose(observed, before + delta, rtol=0.0, atol=1.0e-8):
            raise RuntimeError(f"center-of-mass perturbation did not read back: {path}")
        records.append({
            "rigid_prim_path": str(path),
            "before_center_of_mass_m": before.tolist(),
            "observed_center_of_mass_m": observed.tolist(),
        })
    scene["center_of_mass_audit"] = {
        "applied": requested is not None,
        "requested_delta_object_m": requested,
        "parts": records,
    }


def _initial_same_reset_rgbd_trace(arguments, inputs, grasp, dynamic):
    high_mode = arguments.mode in VISION_MOTION_MODES
    result = {
        "schema_version": (
            "kcg_te_visual_high_reobserve_trace_v1"
            if high_mode
            else "kcg_te_same_reset_rgbd_observe_trace_v1"
        ),
        "object_id": arguments.object_id,
        "candidate_id": grasp["grasp_id"],
        "mode": arguments.mode,
        "capture_id": arguments.capture_id,
        "config_sha256": file_sha256(inputs.config.path),
        "same_reset_rgbd_config": {
            "path": str(
                arguments.same_reset_rgbd_config_path.relative_to(
                    inputs.repository_root
                )
            ),
            "sha256": file_sha256(arguments.same_reset_rgbd_config_path),
        },
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "control_authorized": False,
        "motion_plan_constructed": False,
        "world_pregrasp_target_consumed": False,
        "robot_target_position_velocity_api_call_count": 0,
        "robot_motion_command_count": 0,
        "world_reset_count": 0,
        "object_pose_writes_after_reset": 0,
        "provider_truth_read_before_seal": False,
        "online_object_or_contact_truth_used": False,
        "truth_audit_data_returned_to_controller": False,
    }
    if high_mode:
        result["visual_high_reobserve_config"] = {
            "path": str(
                arguments.visual_high_reobserve_config_path.relative_to(
                    inputs.repository_root
                )
            ),
            "sha256": file_sha256(
                arguments.visual_high_reobserve_config_path
            ),
        }
        result["approach_high_motion_authorized"] = True
        grasp_servo = arguments.mode == VISION_GRASP_SERVO_MODE
        result["descent_authorized"] = grasp_servo
        result["object_contact_authorized"] = grasp_servo
        result["finger_closure_authorized"] = grasp_servo
        result["approach_high_motion_duration"] = {
            "configured_s": float(
                arguments.approach_high_motion_duration_configured_s
            ),
            "effective_s": float(
                arguments.approach_high_motion_duration_effective_s
            ),
            "preshape_duration_s": float(
                dynamic["approach_above_duration_s"]
            ),
            "single_variable_variant_applied": (
                arguments.visual_high_reobserve_variant_document is not None
            ),
        }
        result["dynamic_inertia_compensation"] = {
            "enabled": bool(arguments.dynamic_inertia_compensation_enabled),
            "enabled_phase": "approach_above",
            "algorithm": _CausalFrozenHandDynamicInertia.METHOD,
            "force_and_torque_limits_unchanged": True,
            "object_or_contact_truth_used": False,
        }
        if arguments.dynamic_inertia_compensation_enabled:
            result["dynamic_inertia_offline_replay"] = (
                arguments.dynamic_inertia_offline_replay_audit
            )
    return result


def _initial_trace(arguments, inputs, grasp, motion_plan, dynamic):
    criteria = {
        key: dynamic[key]
        for key in (
            "lift_distance_m", "lift_tolerance_m", "hold_duration_s",
            "table_release_clearance_m", "maximum_table_penetration_m",
            "lift_acceleration_difference_window_samples",
            "lift_acceleration_tolerance_m_s2",
        )
    }
    criteria["sustained_three_contact_samples"] = int(
        dynamic["contact_consecutive_samples"]
    )
    criteria["registered_lift_peak_acceleration_m_s2"] = float(
        grasp["lift_trajectory"]["registered_peak_acceleration_m_s2"]
    )
    criteria["first_finger_diagnostic_duration_s"] = float(dynamic["preload_duration_s"])
    criteria["maximum_finger_target_increment_rad"] = float(dynamic[
        "finger_maximum_speed_rad_s"]) * float(dynamic["physics_dt_s"])
    return {
        "schema_version": "carts_grasp_v2_dynamic_trace_v1",
        "object_id": arguments.object_id, "candidate_id": grasp["grasp_id"],
        "mode": arguments.mode,
        "initialized_at_pregrasp": bool(arguments.initialize_at_pregrasp),
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp": dict(grasp),
        "offline_worst_task_margin": None,
        "offline_task_gate_passed": False,
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "maximum_joint_speed_limit_rad_s": float(
            dynamic["maximum_joint_speed_rad_s"]
        ),
        "object_pose_writes_after_start": 0,
        "controller_online_signals": list(motion_plan["online_signals"]),
        "online_object_or_contact_truth_used": False,
        "truth_audit_data_returned_to_controller": False,
        "visual_transport_target_consumption": (
            arguments.visual_transport_target_consumption
        ),
        "disturbance_executed": (
            arguments.robustness_scenario_name != "nominal"
        ),
        "robustness_scenario": arguments.robustness_scenario_name,
        "robustness_perturbation": arguments.robustness_perturbation,
        "hand_tilt_about_object_pivot_audit": (
            arguments.hand_tilt_about_object_pivot_audit
        ),
        "postgrasp_disturbance": arguments.postgrasp_disturbance,
        "postgrasp_disturbance_execution": {
            "requested": arguments.postgrasp_disturbance is not None,
            "started": False,
            "completed": False,
            "failure_reason": None,
        },
        "configured_preload_increment_rad": (
            arguments.configured_preload_increment_rad
        ),
        "effective_preload_increment_rad": (
            arguments.effective_preload_increment_rad
        ),
        "configured_lift_arm_damping_nm_s_rad": (
            arguments.configured_lift_arm_damping_nm_s_rad
        ),
        "effective_lift_arm_damping_nm_s_rad": (
            arguments.effective_lift_arm_damping_nm_s_rad
        ),
        "configured_finger_preload_scales": (
            arguments.configured_finger_preload_scales
        ),
        "effective_finger_preload_scales": (
            arguments.effective_finger_preload_scales
        ),
        "configured_lift_arm_damping_nm_s_rad": (
            arguments.configured_lift_arm_damping_nm_s_rad
        ),
        "effective_lift_arm_damping_nm_s_rad": (
            arguments.effective_lift_arm_damping_nm_s_rad
        ),
        "required_closing_joint_effort_nm": (
            arguments.required_closing_joint_effort_nm
        ),
        "predicted_unit_task_closing_joint_effort_nm": (
            arguments.required_closing_joint_effort_nm
        ),
        "prelift_contact_confirmation_effort_nm": (
            arguments.dynamic_settings["contact_effort_rise_nm"]
        ),
        "effort_regulation_tolerance_nm": (
            arguments.dynamic_settings["effort_regulation_tolerance_nm"]
        ),
        "configured_closing_order": arguments.configured_closing_order,
        "effective_closing_order": arguments.effective_closing_order,
        "configured_contact_coordination_mode": (
            arguments.configured_contact_coordination_mode
        ),
        "effective_contact_coordination_mode": (
            arguments.effective_contact_coordination_mode
        ),
        "configured_palm_joint_position_rad": (
            arguments.configured_palm_joint_position_rad
        ),
        "effective_palm_joint_position_rad": (
            arguments.effective_palm_joint_position_rad
        ),
        "configured_approach_high_seed_arm_positions_rad": (
            arguments.configured_approach_high_seed_arm_positions_rad
        ),
        "effective_approach_high_seed_arm_positions_rad": (
            arguments.effective_approach_high_seed_arm_positions_rad
        ),
        "motion_plan": motion_plan,
        "criteria": criteria,
        "controller_outcome": {"completed": False, "failure_reason": None},
        "samples": [],
    }


def _initial_isolated_trace(arguments, inputs, grasp, motion_plan, dynamic):
    reference_path = Path(arguments.reference_trace).resolve()
    reference = arguments.reference_document
    observed = (reference.get("object_id"), reference.get("candidate_id"))
    config_sha256 = file_sha256(inputs.config.path)
    if (observed != (arguments.object_id, grasp["grasp_id"])
            or reference.get("mode") not in ("preflight", "first-finger-diagnostic", "grasp-lift")
            or reference.get("config_sha256") != config_sha256):
        raise ValueError("isolated diagnostic reference differs from the failed run")
    if (
        float(reference.get("physics_dt_s", -1.0)) != float(dynamic["physics_dt_s"])
        or _json_sha256(reference.get("motion_plan")) != _json_sha256(motion_plan)
    ):
        raise ValueError("isolated diagnostic trajectory differs from the failed run")
    return {
        "schema_version": "carts_grasp_v2_isolated_hand_diagnostic_v1",
        "object_id": arguments.object_id, "candidate_id": grasp["grasp_id"],
        "mode": arguments.mode,
        "config_sha256": config_sha256,
        "physics_dt_s": float(dynamic["physics_dt_s"]),
        "maximum_joint_speed_limit_rad_s": float(
            dynamic["maximum_joint_speed_rad_s"]
        ),
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "object_loaded": False, "table_loaded": False,
        "online_object_or_contact_truth_used": False,
        "object_pose_writes_after_start": 0,
        "reference_trace": str(reference_path),
        "reference_trace_sha256": file_sha256(reference_path),
        "reference_failure_reason": reference["controller_outcome"]["failure_reason"],
        "reference_maximum_joint_speed_rad_s": reference["controller_outcome"][
            "maximum_joint_speed_rad_s"],
        "reference_controller_source_sha256": reference["evidence_binding"][
            "controller_source_sha256"],
        "reference_robot_asset_sha256": reference["evidence_binding"][
            "robot_asset_sha256"],
        "reference_active_drive_audit_sha256": _json_sha256(
            reference["controller_outcome"]["native_drive_audit"]),
        "motion_plan": motion_plan,
        "samples": [],
    }


def _full_drive_audit(robot, dof_names):
    stiffnesses, dampings = robot.get_dof_gains(indices=0)
    efforts = robot.get_dof_max_efforts(indices=0)
    drive_types = robot.get_dof_drive_types(indices=0)[0]
    stiffnesses = stiffnesses.numpy()[0]
    dampings = dampings.numpy()[0]
    efforts = efforts.numpy()[0]
    return {
        name: {
            "drive_type": drive_types[index],
            "stiffness": float(stiffnesses[index]),
            "damping": float(dampings[index]),
            "maximum_effort_nm": float(efforts[index]),
            "mimic_source": MIMIC_HAND_JOINTS.get(name),
        }
        for index, name in enumerate(dof_names)
    }


def _isolated_gravity(repository, scene_entry):
    from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene

    key = (
        "scene_config" if scene_entry["scene_kind"] == "D38999_PAIR_TABLETOP"
        else "environment_scene_config"
    )
    scene_path = (repository / scene_entry[key]).resolve()
    return float(load_d38999_tabletop_scene(scene_path).physics.gravity_m_s2), scene_path


def _create_isolated_runtime(repository, arguments, inputs, scene_entry, trace):
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    dynamic = arguments.dynamic_settings
    robot_asset = arguments.robot_asset_path
    gravity, gravity_source = _isolated_gravity(repository, scene_entry)
    World.clear_instance()
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        stage_units_in_meters=1.0, physics_dt=float(dynamic["physics_dt_s"]),
        rendering_dt=1.0 / 60.0, backend="numpy", device="cuda:0",
        sim_params={"use_gpu_pipeline": True},
    )
    context = world.get_physics_context()
    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    context.set_gravity(gravity)
    world.reset()
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH, EXPECTED_DOF_NAMES, dynamic)
    robot, active_indices, arm_indices, lower, upper, active_audit = robot_data
    backend = gpu_backend_record(world, context)
    if not backend["pass"]:
        raise RuntimeError(f"GPU physics backend audit failed: {backend}")
    trace["physics_backend"] = backend
    trace["gravity_m_s2"] = gravity
    trace["gravity_source"] = str(gravity_source)
    trace["robot_asset"] = str(robot_asset)
    trace["robot_asset_sha256"] = file_sha256(robot_asset)
    trace["active_drive_audit"] = active_audit
    if (
        trace["robot_asset_sha256"] != trace["reference_robot_asset_sha256"]
        or _json_sha256(active_audit) != trace["reference_active_drive_audit_sha256"]
    ):
        raise RuntimeError("isolated robot asset or active drive differs from reference")
    trace["all_dof_drive_audit"] = _full_drive_audit(robot, robot.dof_names)
    trace["initial_joint_audit"] = audit_initial_joint_state(robot, robot.dof_names)
    trace["mimic_schema_audit"] = audit_mimic_schema(
        get_current_stage(), ROBOT_ROOT, MIMIC_HAND_JOINTS
    )
    recorder = IsolatedHandRecorder(
        robot=robot, dof_names=robot.dof_names,
        active_names=control.ARM_JOINT_NAMES + control.ACTIVE_HAND_JOINT_NAMES,
        hand_names=EXPECTED_DOF_NAMES[7:], physics_dt_s=dynamic["physics_dt_s"],
        drive_settings=dynamic,
    )
    return world, recorder, robot_data


def _execute_isolated(repository, arguments, output, inputs, scene_entry, motion_plan, trace):
    dynamic = arguments.dynamic_settings
    world, recorder, robot_data = _create_isolated_runtime(
        repository, arguments, inputs, scene_entry, trace
    )
    robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
    stepper = control.JointSignalStepper(
        robot=robot, world=world, auditor=recorder, active_indices=active_indices,
        arm_indices=arm_indices, arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui)
    pregrasp = control.run_pregrasp_sequence(stepper, motion_plan, dynamic)
    if arguments.reference_document.get("mode") in ("first-finger-diagnostic", "grasp-lift"):
        for row in arguments.reference_document["samples"][stepper.step_index:]:
            target = np.asarray(row["active_targets_rad"], dtype=np.float64)
            stepper.advance(f"replay_{row['phase']}", target[:7], target[7:])
            if stepper.abort_reason is not None:
                break
    outcome = control.controller_outcome(
        stepper, mode="preflight", native_drive_audit=drive_audit,
        pregrasp=pregrasp,
        grasp={"contact_controller": None, "failure_reason": stepper.abort_reason})
    trace["samples"] = recorder.samples
    trace["controller_outcome"] = outcome
    trace["reference_target_comparison"] = compare_reference_targets(
        arguments.reference_document, trace["samples"])
    trace["runtime"] = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            check=True, capture_output=True,
        ).stdout.strip(),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "controller_source_sha256": file_sha256(Path(__file__).with_name("controller.py")),
    }
    metrics = evaluate_isolated_hand_trace(trace)
    (output / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["diagnostic_pass"] else 2


def _evidence_binding(repository, arguments, inputs, grasp, scene, robot_asset):
    object_asset = scene["object_asset"]
    evidence_paths = tuple(scene["evidence_paths"])
    binding = {
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp_sha256": _json_sha256(grasp),
        "control_plan_sha256": _json_sha256(grasp["control_plan"]),
        "runtime_resources_sha256": file_sha256(arguments.runtime_resources_path),
        "capacity_audit_sha256": arguments.runtime_resources_document[
            "capacity_audit_sha256"],
        "scene_evidence_sha256": {
            str(path.relative_to(repository)): file_sha256(path) for path in evidence_paths
        },
        "environment_scope": scene["environment_scope"],
        "robustness_scenario": arguments.robustness_scenario_name,
        "robustness_perturbation": arguments.robustness_perturbation,
        "effective_finger_preload_scales": (
            arguments.effective_finger_preload_scales
        ),
        "required_closing_joint_effort_nm": (
            arguments.required_closing_joint_effort_nm
        ),
        "predicted_unit_task_closing_joint_effort_nm": (
            arguments.required_closing_joint_effort_nm
        ),
        "prelift_contact_confirmation_effort_nm": (
            arguments.dynamic_settings["contact_effort_rise_nm"]
        ),
        "effort_regulation_tolerance_nm": (
            arguments.dynamic_settings["effort_regulation_tolerance_nm"]
        ),
        "configured_closing_order": arguments.configured_closing_order,
        "effective_closing_order": arguments.effective_closing_order,
        "configured_contact_coordination_mode": (
            arguments.configured_contact_coordination_mode
        ),
        "effective_contact_coordination_mode": (
            arguments.effective_contact_coordination_mode
        ),
        "applied_initial_object_translation_delta_m": scene[
            "applied_initial_object_translation_delta_m"
        ],
        "applied_initial_object_yaw_delta_rad": scene[
            "applied_initial_object_yaw_delta_rad"
        ],
        "contact_friction_material_audit": scene[
            "contact_friction_material_audit"
        ],
        "object_mass_audit": scene["object_mass_audit"],
        "center_of_mass_audit": scene["center_of_mass_audit"],
        "finger_joint_target_audit": arguments.finger_joint_target_audit,
        "hand_tilt_about_object_pivot_audit": (
            arguments.hand_tilt_about_object_pivot_audit
        ),
        "object_asset_sha256": file_sha256(object_asset),
        "robot_asset_sha256": file_sha256(robot_asset),
        "controller_source_sha256": file_sha256(Path(__file__).with_name("controller.py")),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "evaluator_source_sha256": file_sha256(Path(__file__).with_name("evaluate_run.py")),
        "engine_health_source_sha256": file_sha256(Path(__file__).with_name("engine_health.py")),
    }
    if arguments.visual_transport_target_binding is not None:
        binding["visual_transport_target"] = (
            arguments.visual_transport_target_binding
        )
    if arguments.mode in (SAME_RESET_RGBD_MODE, *VISION_MOTION_MODES):
        binding["same_reset_rgbd_observe"] = {
            "config_sha256": file_sha256(
                arguments.same_reset_rgbd_config_path
            ),
            "frozen_source_sha256": {
                name: file_sha256(path)
                for name, path in arguments.same_reset_rgbd_sources.items()
            },
            "provider_scope": "TRANSPORT_PLUG_ONLY",
            "robot_motion_authorized": arguments.mode in VISION_MOTION_MODES,
        }
    if arguments.mode in VISION_MOTION_MODES:
        prefix_document = arguments.visual_second_short_prefix_document
        short_prefix = prefix_document is not None
        binding["visual_high_reobserve"] = {
            "config_sha256": file_sha256(
                arguments.visual_high_reobserve_config_path
            ),
            "fresh_provider_must_precede_motion_plan": True,
            "motion_scope": (
                "RGBD_SERVO_THEN_GRASP_LIFT"
                if arguments.mode == VISION_GRASP_SERVO_MODE
                else
                "APPROACH_HIGH_THEN_SECOND_VISUAL_FREE_SPACE_PREFIX"
                if short_prefix
                else "APPROACH_HIGH_ONLY"
            ),
            "second_visual_short_prefix_authorized": short_prefix,
            "second_visual_target_clearance_from_pregrasp_m": (
                None
                if prefix_document is None
                else float(
                    prefix_document["short_prefix"][
                        "target_clearance_from_pregrasp_m"
                    ]
                )
            ),
            "second_visual_target_method": (
                None
                if prefix_document is None
                else prefix_document["short_prefix"].get(
                    "target_method", "PLANNER_WAYPOINT_INDEX"
                )
            ),
            "descent_contact_and_closure_authorized": False,
            "online_object_semantic_instance_or_contact_truth_used": False,
        }
    return binding


def _write_json_sealed(path: Path, document: Mapping[str, object]) -> None:
    payload = json.dumps(
        document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _quaternion_wxyz_rotation(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("posthoc object quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise RuntimeError("posthoc object quaternion is degenerate")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _load_frozen_hand_inertials(path: Path) -> tuple[dict[str, Any], ...]:
    root = ET.parse(path).getroot()
    records: list[dict[str, Any]] = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_element = inertial.find("mass")
        origin_element = inertial.find("origin")
        inertia_element = inertial.find("inertia")
        if (
            mass_element is None
            or origin_element is None
            or inertia_element is None
        ):
            raise ValueError("hand inertial lacks mass, origin, or inertia")
        rpy = np.fromstring(origin_element.get("rpy", "0 0 0"), sep=" ")
        xyz = np.fromstring(origin_element.get("xyz", ""), sep=" ")
        mass = float(mass_element.get("value", "nan"))
        try:
            ixx = float(inertia_element.attrib["ixx"])
            ixy = float(inertia_element.attrib["ixy"])
            ixz = float(inertia_element.attrib["ixz"])
            iyy = float(inertia_element.attrib["iyy"])
            iyz = float(inertia_element.attrib["iyz"])
            izz = float(inertia_element.attrib["izz"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("hand inertia tensor is incomplete") from error
        inertia_tensor = np.asarray(
            (
                (ixx, ixy, ixz),
                (ixy, iyy, iyz),
                (ixz, iyz, izz),
            ),
            dtype=np.float64,
        )
        if (
            xyz.shape != (3,)
            or rpy.shape != (3,)
            or not np.all(np.isfinite(xyz))
            or not np.allclose(rpy, 0.0, rtol=0.0, atol=0.0)
            or not math.isfinite(mass)
            or mass <= 0.0
            or not np.all(np.isfinite(inertia_tensor))
            or not np.allclose(
                inertia_tensor,
                inertia_tensor.T,
                rtol=0.0,
                atol=0.0,
            )
            or float(np.min(np.linalg.eigvalsh(inertia_tensor))) <= 0.0
        ):
            raise ValueError("hand inertial mass/COM/tensor contract differs")
        records.append(
            {
                "link_name": str(link.attrib["name"]),
                "mass_kg": mass,
                "center_of_mass_link_m": xyz,
                "inertia_about_com_link_kg_m2": inertia_tensor,
            }
        )
    if (
        {record["link_name"] for record in records}
        != {
            "handbase_link",
            "f1Link1",
            "f1Link2",
            "f1Link3",
            "f2Link1",
            "f2Link2",
            "f3Link1",
            "f3Link2",
            "f3Link3",
        }
        or not math.isclose(
            sum(float(record["mass_kg"]) for record in records),
            2.091942,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("frozen hand inertial link set or total mass differs")
    return tuple(records)


def _aggregate_frozen_hand_inertia(
    robot_model: Any,
    hand_inertials: Sequence[Mapping[str, Any]],
    frozen_hand_positions: Sequence[float],
) -> dict[str, Any]:
    hand = np.asarray(frozen_hand_positions, dtype=np.float64)
    if hand.shape != (4,) or not np.all(np.isfinite(hand)):
        raise ValueError("frozen pregrasp hand state must contain four values")
    links = robot_model.forward_kinematics(
        tuple(np.concatenate((np.zeros(7, dtype=np.float64), hand))),
        enforce_limits=False,
    )
    sensor = np.asarray(links["handbase_link"], dtype=np.float64)
    sensor_from_world = np.linalg.inv(sensor)
    transformed: list[tuple[float, np.ndarray, np.ndarray]] = []
    for record in hand_inertials:
        link = np.asarray(links[str(record["link_name"])], dtype=np.float64)
        sensor_from_link = sensor_from_world @ link
        center_link = np.asarray(
            record["center_of_mass_link_m"], dtype=np.float64
        )
        center_sensor = (
            sensor_from_link @ np.concatenate((center_link, (1.0,)))
        )[:3]
        rotation_sensor_link = sensor_from_link[:3, :3]
        inertia_sensor = (
            rotation_sensor_link
            @ np.asarray(
                record["inertia_about_com_link_kg_m2"], dtype=np.float64
            )
            @ rotation_sensor_link.T
        )
        transformed.append(
            (float(record["mass_kg"]), center_sensor, inertia_sensor)
        )
    mass = float(sum(row[0] for row in transformed))
    center = sum(
        (row[0] * row[1] for row in transformed),
        start=np.zeros(3, dtype=np.float64),
    ) / mass
    inertia = np.zeros((3, 3), dtype=np.float64)
    for link_mass, link_center, link_inertia in transformed:
        offset = link_center - center
        inertia += link_inertia + link_mass * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
    if (
        not math.isclose(mass, 2.091942, rel_tol=0.0, abs_tol=1.0e-12)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(inertia))
        or float(np.min(np.linalg.eigvalsh(inertia))) <= 0.0
    ):
        raise ValueError("frozen aggregate hand inertia is invalid")
    return {
        "mass_kg": mass,
        "center_of_mass_sensor_m": center,
        "inertia_about_com_sensor_kg_m2": inertia,
        "frozen_hand_positions_rad": hand,
        "source_link_count": len(transformed),
    }


class _CausalFrozenHandDynamicInertia:
    """Predict hand-on-arm inertia from joint state without force feedback."""

    METHOD = (
        "FROZEN_PREGRASP_9_LINK_AGGREGATE_JACOBIAN_BACKWARD_DIFFERENCE_V1"
    )

    def __init__(
        self,
        *,
        robot_model: Any,
        aggregate: Mapping[str, Any],
        physics_dt_s: float,
    ) -> None:
        self.robot_model = robot_model
        self.mass_kg = float(aggregate["mass_kg"])
        self.center_sensor = np.asarray(
            aggregate["center_of_mass_sensor_m"], dtype=np.float64
        )
        self.inertia_sensor = np.asarray(
            aggregate["inertia_about_com_sensor_kg_m2"], dtype=np.float64
        )
        self.physics_dt_s = float(physics_dt_s)
        self.previous_step: int | None = None
        self.previous_velocity: np.ndarray | None = None
        self.previous_jacobian: np.ndarray | None = None
        if (
            self.mass_kg <= 0.0
            or self.center_sensor.shape != (3,)
            or self.inertia_sensor.shape != (3, 3)
            or self.physics_dt_s <= 0.0
            or not np.all(np.isfinite(self.center_sensor))
            or not np.all(np.isfinite(self.inertia_sensor))
        ):
            raise ValueError("causal dynamic inertia model is invalid")

    def observe(
        self,
        *,
        step: int,
        active_positions: np.ndarray,
        active_velocities: np.ndarray,
        sensor_world: np.ndarray,
    ) -> dict[str, Any]:
        positions = np.asarray(active_positions, dtype=np.float64)
        velocities = np.asarray(active_velocities, dtype=np.float64)
        sensor = np.asarray(sensor_world, dtype=np.float64)
        jacobian_positions = positions.copy()
        jacobian_positions[np.abs(jacobian_positions) < 1.0e-6] = 0.0
        jacobian = np.asarray(
            self.robot_model.geometric_jacobian(
                "handbase_link", tuple(jacobian_positions)
            ),
            dtype=np.float64,
        )
        if (
            positions.shape != (11,)
            or velocities.shape != (11,)
            or jacobian.shape != (6, 11)
            or sensor.shape != (4, 4)
            or not all(
                np.all(np.isfinite(value))
                for value in (positions, velocities, jacobian, sensor)
            )
        ):
            raise RuntimeError("dynamic inertia kinematic input is invalid")
        consecutive = (
            self.previous_step is not None
            and int(step) == int(self.previous_step) + 1
        )
        ready = bool(
            consecutive
            and self.previous_velocity is not None
            and self.previous_jacobian is not None
        )
        predicted_sensor = None
        if ready:
            joint_acceleration = (
                velocities - self.previous_velocity
            ) / self.physics_dt_s
            jacobian_rate = (
                jacobian - self.previous_jacobian
            ) / self.physics_dt_s
            twist = jacobian @ velocities
            spatial_acceleration = (
                jacobian @ joint_acceleration + jacobian_rate @ velocities
            )
            angular_velocity_world = twist[3:]
            sensor_linear_acceleration_world = spatial_acceleration[:3]
            angular_acceleration_world = spatial_acceleration[3:]
            rotation_world_sensor = sensor[:3, :3]
            center_offset_world = rotation_world_sensor @ self.center_sensor
            center_acceleration_world = (
                sensor_linear_acceleration_world
                + np.cross(angular_acceleration_world, center_offset_world)
                + np.cross(
                    angular_velocity_world,
                    np.cross(angular_velocity_world, center_offset_world),
                )
            )
            inertia_world = (
                rotation_world_sensor
                @ self.inertia_sensor
                @ rotation_world_sensor.T
            )
            force_world = self.mass_kg * center_acceleration_world
            torque_world = (
                np.cross(center_offset_world, force_world)
                + inertia_world @ angular_acceleration_world
                + np.cross(
                    angular_velocity_world,
                    inertia_world @ angular_velocity_world,
                )
            )
            predicted_sensor = np.concatenate(
                (
                    rotation_world_sensor.T @ force_world,
                    rotation_world_sensor.T @ torque_world,
                )
            )
            if not np.all(np.isfinite(predicted_sensor)):
                raise RuntimeError("dynamic inertia prediction is nonfinite")
        self.previous_step = int(step)
        self.previous_velocity = velocities.copy()
        self.previous_jacobian = jacobian.copy()
        return {
            "method": self.METHOD,
            "ready": ready,
            "reason": (
                "CURRENT_AND_PREVIOUS_CAUSAL_STATE_AVAILABLE"
                if ready
                else "WAITING_FOR_ONE_CONSECUTIVE_PREVIOUS_STATE"
            ),
            "predicted_positive_inertia_wrench_sensor": predicted_sensor,
            "force_balance": (
                "canonical=gravity+external_contact-inertia; "
                "external_residual=gravity_residual+predicted_positive_inertia"
            ),
            "physical_wrench_used_as_prediction_input": False,
            "object_or_contact_truth_used": False,
        }


def replay_high_observation_dynamic_inertia(
    *,
    samples: Sequence[Mapping[str, Any]],
    robot_model: Any,
    hand_inertials: Sequence[Mapping[str, Any]],
    frozen_hand_positions: Sequence[float],
    physics_dt_s: float,
    enabled_phase: str = "approach_above",
) -> dict[str, Any]:
    """Replay the exact online predictor without Isaac or truth inputs."""

    aggregate = _aggregate_frozen_hand_inertia(
        robot_model, hand_inertials, frozen_hand_positions
    )
    predictor = _CausalFrozenHandDynamicInertia(
        robot_model=robot_model,
        aggregate=aggregate,
        physics_dt_s=physics_dt_s,
    )
    old_force: list[float] = []
    old_torque: list[float] = []
    new_force: list[float] = []
    new_torque: list[float] = []
    old_enabled_force: list[float] = []
    old_enabled_torque: list[float] = []
    new_enabled_force: list[float] = []
    new_enabled_torque: list[float] = []
    enabled_not_ready = 0
    applied_count = 0
    for sample in samples:
        positions = np.asarray(sample["active_positions_rad"], dtype=np.float64)
        velocities = np.asarray(
            sample["active_velocities_rad_s"], dtype=np.float64
        )
        links = robot_model.forward_kinematics(
            tuple(positions), enforce_limits=False
        )
        sensor = np.asarray(links["handbase_link"], dtype=np.float64)
        prediction = predictor.observe(
            step=int(sample["step"]),
            active_positions=positions,
            active_velocities=velocities,
            sensor_world=sensor,
        )
        if sample.get("run_specific_tare_canonical_sensor_wrench") is None:
            continue
        old_sensor = (
            np.asarray(
                sample["hand2arm_canonical_sensor_wrench"], dtype=np.float64
            )
            - np.asarray(
                sample["run_specific_tare_canonical_sensor_wrench"],
                dtype=np.float64,
            )
            - (
                np.asarray(
                    sample["modeled_hand_gravity_sensor_wrench"],
                    dtype=np.float64,
                )
                - np.asarray(
                    sample["run_specific_tare_modeled_gravity_sensor_wrench"],
                    dtype=np.float64,
                )
            )
        )
        new_sensor = old_sensor.copy()
        if str(sample["phase"]) == enabled_phase:
            if prediction["ready"] is not True:
                enabled_not_ready += 1
            else:
                predicted_sensor = prediction[
                    "predicted_positive_inertia_wrench_sensor"
                ]
                new_sensor += predicted_sensor
                applied_count += 1
        old_force.append(float(np.linalg.norm(old_sensor[:3])))
        old_torque.append(float(np.linalg.norm(old_sensor[3:])))
        new_force.append(float(np.linalg.norm(new_sensor[:3])))
        new_torque.append(float(np.linalg.norm(new_sensor[3:])))
        if str(sample["phase"]) == enabled_phase:
            old_enabled_force.append(old_force[-1])
            old_enabled_torque.append(old_torque[-1])
            new_enabled_force.append(new_force[-1])
            new_enabled_torque.append(new_torque[-1])
    return {
        "schema_version": "kcg_te_high_dynamic_inertia_offline_replay_v1",
        "algorithm": _CausalFrozenHandDynamicInertia.METHOD,
        "enabled_phase": enabled_phase,
        "sample_count": len(samples),
        "applied_sample_count": applied_count,
        "enabled_phase_not_ready_count": enabled_not_ready,
        "aggregate": _json_ready(aggregate),
        "old_full_record_peak_force_n": max(old_force),
        "old_full_record_peak_torque_nm": max(old_torque),
        "new_full_record_peak_force_n": max(new_force),
        "new_full_record_peak_torque_nm": max(new_torque),
        "old_enabled_phase_peak_force_n": max(old_enabled_force),
        "old_enabled_phase_peak_torque_nm": max(old_enabled_torque),
        "new_enabled_phase_peak_force_n": max(new_enabled_force),
        "new_enabled_phase_peak_torque_nm": max(new_enabled_torque),
        "physical_wrench_used_as_prediction_input": False,
        "object_semantic_instance_or_contact_truth_used": False,
    }


class _HighObservationWristFtAuditor:
    """Truth-free joint and physical hand2arm safety recorder."""

    _PLANNED_CONTACT_PHASES = frozenset(
        ("preload", "prelift_effort_check", "lift", "hold")
    )
    _PLANNED_CONTACT_PHASE_PREFIXES = (
        "parallel_contact_",
        "finger_",
        "postgrasp_disturbance_",
    )

    def __init__(
        self,
        *,
        ft_articulation: Any,
        reaction_row: int,
        robot_model: Any,
        hand_inertials: Sequence[Mapping[str, Any]],
        gravity_m_s2: float,
        physics_dt_s: float,
        task_rotation_world: np.ndarray,
        force_limit_n: float,
        torque_limit_nm: float,
        planned_contact_torque_limit_nm: float | None = None,
        dynamic_inertia_compensation_enabled: bool = False,
        frozen_hand_positions: Sequence[float] | None = None,
        dynamic_inertia_enabled_phases: Sequence[str] = ("approach_above",),
    ) -> None:
        self.ft_articulation = ft_articulation
        self.reaction_row = int(reaction_row)
        self.robot_model = robot_model
        self.hand_inertials = tuple(hand_inertials)
        self.gravity_m_s2 = abs(float(gravity_m_s2))
        self.physics_dt_s = float(physics_dt_s)
        self.task_rotation_world = np.asarray(
            task_rotation_world, dtype=np.float64
        )
        self.force_limit_n = float(force_limit_n)
        self.torque_limit_nm = float(torque_limit_nm)
        self.planned_contact_torque_limit_nm = (
            None
            if planned_contact_torque_limit_nm is None
            else float(planned_contact_torque_limit_nm)
        )
        self.dynamic_inertia_compensation_enabled = bool(
            dynamic_inertia_compensation_enabled
        )
        self.dynamic_inertia_enabled_phases = frozenset(
            str(phase) for phase in dynamic_inertia_enabled_phases
        )
        if (
            self.task_rotation_world.shape != (3, 3)
            or not np.all(np.isfinite(self.task_rotation_world))
            or not np.allclose(
                self.task_rotation_world.T @ self.task_rotation_world,
                np.eye(3),
                rtol=0.0,
                atol=1.0e-9,
            )
            or min(
                self.gravity_m_s2,
                self.physics_dt_s,
                self.force_limit_n,
                self.torque_limit_nm,
            )
            <= 0.0
            or not self.dynamic_inertia_enabled_phases
            or any(not phase for phase in self.dynamic_inertia_enabled_phases)
            or (
                self.planned_contact_torque_limit_nm is not None
                and (
                    not math.isfinite(self.planned_contact_torque_limit_nm)
                    or self.planned_contact_torque_limit_nm
                    < self.torque_limit_nm
                )
            )
        ):
            raise ValueError("high-observation FT recorder contract is invalid")
        self.stepper = None
        self.samples: list[dict[str, Any]] = []
        self._tare_sensor_samples: list[np.ndarray] = []
        self._tare_gravity_samples: list[np.ndarray] = []
        self.tare_canonical_sensor: np.ndarray | None = None
        self.tare_gravity_sensor: np.ndarray | None = None
        self.tare_model_residual_sensor: np.ndarray | None = None
        self.gate_enabled = False
        self.first_ft_stop: dict[str, Any] | None = None
        self.precontact_recovery_active = False
        self.precontact_recovery_force_ceiling_n: float | None = None
        self.precontact_recovery_torque_ceiling_nm: float | None = None
        self.precontact_recovery_worsened = False
        self.maximum_residual_force_n = 0.0
        self.maximum_residual_torque_nm = 0.0
        self.phase_counts: dict[str, int] = {}
        self.dynamic_inertia_aggregate = None
        self.dynamic_inertia_predictor = None
        if self.dynamic_inertia_compensation_enabled:
            if frozen_hand_positions is None:
                raise ValueError(
                    "dynamic inertia compensation requires frozen hand positions"
                )
            self.dynamic_inertia_aggregate = _aggregate_frozen_hand_inertia(
                self.robot_model,
                self.hand_inertials,
                frozen_hand_positions,
            )
            self.dynamic_inertia_predictor = _CausalFrozenHandDynamicInertia(
                robot_model=self.robot_model,
                aggregate=self.dynamic_inertia_aggregate,
                physics_dt_s=self.physics_dt_s,
            )

    def _kinematics(
        self, active_positions: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], np.ndarray]:
        links = self.robot_model.forward_kinematics(
            tuple(map(float, active_positions)), enforce_limits=False
        )
        sensor = np.asarray(links["handbase_link"], dtype=np.float64)
        if sensor.shape != (4, 4) or not np.all(np.isfinite(sensor)):
            raise RuntimeError("handbase FK is invalid during FT monitoring")
        return links, sensor

    def _gravity_wrench_sensor(
        self, links: Mapping[str, np.ndarray], sensor: np.ndarray
    ) -> np.ndarray:
        force_world_total = np.zeros(3, dtype=np.float64)
        torque_world_total = np.zeros(3, dtype=np.float64)
        sensor_position = sensor[:3, 3]
        for record in self.hand_inertials:
            transform = np.asarray(
                links[str(record["link_name"])], dtype=np.float64
            )
            local_com = np.asarray(
                record["center_of_mass_link_m"], dtype=np.float64
            )
            com_world = (
                transform
                @ np.concatenate((local_com, np.asarray((1.0,))))
            )[:3]
            force_world = np.asarray(
                (0.0, 0.0, -float(record["mass_kg"]) * self.gravity_m_s2),
                dtype=np.float64,
            )
            force_world_total += force_world
            torque_world_total += np.cross(
                com_world - sensor_position, force_world
            )
        rotation_sensor_world = sensor[:3, :3].T
        return np.concatenate(
            (
                rotation_sensor_world @ force_world_total,
                rotation_sensor_world @ torque_world_total,
            )
        )

    def _read_canonical_sensor(self) -> tuple[np.ndarray, np.ndarray]:
        values = _host_array(self.ft_articulation.get_measured_joint_forces())
        if (
            values.ndim != 2
            or values.shape[1] != 6
            or self.reaction_row < 0
            or self.reaction_row >= len(values)
            or not np.all(np.isfinite(values))
        ):
            raise RuntimeError(
                f"unexpected physical hand2arm reaction table: {values.shape}"
            )
        raw = values[self.reaction_row].copy()
        return raw, -raw

    def finalize_tare(self, minimum_samples: int) -> None:
        if (
            len(self._tare_sensor_samples) < int(minimum_samples)
            or len(self._tare_sensor_samples) != len(self._tare_gravity_samples)
        ):
            raise RuntimeError("insufficient run-specific free-space FT tare")
        sensor = np.stack(self._tare_sensor_samples)
        gravity = np.stack(self._tare_gravity_samples)
        if not np.all(np.isfinite(sensor)) or not np.all(np.isfinite(gravity)):
            raise RuntimeError("run-specific FT tare is nonfinite")
        self.tare_canonical_sensor = np.mean(sensor, axis=0)
        self.tare_gravity_sensor = np.mean(gravity, axis=0)
        self.tare_model_residual_sensor = (
            self.tare_canonical_sensor - self.tare_gravity_sensor
        )
        if (
            float(np.linalg.norm(self.tare_model_residual_sensor[:3]))
            > self.force_limit_n
            or float(np.linalg.norm(self.tare_model_residual_sensor[3:]))
            > self.torque_limit_nm
        ):
            raise RuntimeError(
                "run-specific hand2arm tare disagrees with the frozen hand "
                "gravity model beyond the research soft limit"
            )
        self.gate_enabled = True

    def begin_precontact_recovery(self) -> None:
        if self.first_ft_stop is None:
            raise RuntimeError("FT recovery requires one recorded safety stop")
        self.precontact_recovery_force_ceiling_n = float(
            self.first_ft_stop["resultant_force_n"]
        )
        self.precontact_recovery_torque_ceiling_nm = float(
            self.first_ft_stop["resultant_torque_nm"]
        )
        self.precontact_recovery_active = True

    def capture(
        self,
        *,
        step: int,
        phase: str,
        active_positions: Sequence[float],
        active_velocities: Sequence[float],
        active_efforts: Sequence[float],
        active_targets: Sequence[float],
        arm_control: Mapping[str, Any],
    ) -> None:
        positions = np.asarray(active_positions, dtype=np.float64)
        velocities = np.asarray(active_velocities, dtype=np.float64)
        efforts = np.asarray(active_efforts, dtype=np.float64)
        targets = np.asarray(active_targets, dtype=np.float64)
        links, sensor = self._kinematics(positions)
        dynamic_prediction = (
            None
            if self.dynamic_inertia_predictor is None
            else self.dynamic_inertia_predictor.observe(
                step=int(step),
                active_positions=positions,
                active_velocities=velocities,
                sensor_world=sensor,
            )
        )
        raw_sensor, canonical_sensor = self._read_canonical_sensor()
        gravity_sensor = self._gravity_wrench_sensor(links, sensor)
        if phase == "ft_free_space_tare":
            self._tare_sensor_samples.append(canonical_sensor.copy())
            self._tare_gravity_samples.append(gravity_sensor.copy())
        residual_sensor = None
        gravity_residual_task = None
        predicted_inertia_task = None
        residual_task = None
        dynamic_inertia_applied = False
        force_norm = None
        torque_norm = None
        limit_reason = None
        if (
            self.tare_canonical_sensor is not None
            and self.tare_gravity_sensor is not None
        ):
            planned_contact_mode = bool(
                self.planned_contact_torque_limit_nm is not None
                and (
                    phase in self._PLANNED_CONTACT_PHASES
                    or phase.startswith(
                        self._PLANNED_CONTACT_PHASE_PREFIXES
                    )
                )
            )
            effective_torque_limit_nm = (
                float(self.planned_contact_torque_limit_nm)
                if planned_contact_mode
                else self.torque_limit_nm
            )
            residual_sensor = (
                canonical_sensor
                - self.tare_canonical_sensor
                - (gravity_sensor - self.tare_gravity_sensor)
            )
            gravity_residual_task = transform_wrench_to_task(
                residual_sensor,
                sensor[:3, 3],
                sensor[:3, :3],
                sensor[:3, 3],
                self.task_rotation_world,
            )
            if (
                self.dynamic_inertia_compensation_enabled
                and phase in self.dynamic_inertia_enabled_phases
            ):
                if dynamic_prediction is None or dynamic_prediction["ready"] is not True:
                    limit_reason = "WRIST_FT_DYNAMIC_INERTIA_NOT_READY_ABORT"
                else:
                    predicted_inertia_sensor = dynamic_prediction[
                        "predicted_positive_inertia_wrench_sensor"
                    ]
                    predicted_inertia_task = transform_wrench_to_task(
                        predicted_inertia_sensor,
                        sensor[:3, 3],
                        sensor[:3, :3],
                        sensor[:3, 3],
                        self.task_rotation_world,
                    )
                    residual_sensor = residual_sensor + predicted_inertia_sensor
                    dynamic_inertia_applied = True
            residual_task = transform_wrench_to_task(
                residual_sensor,
                sensor[:3, 3],
                sensor[:3, :3],
                sensor[:3, 3],
                self.task_rotation_world,
            )
            force_norm = float(np.linalg.norm(residual_task[:3]))
            torque_norm = float(np.linalg.norm(residual_task[3:]))
            self.maximum_residual_force_n = max(
                self.maximum_residual_force_n, force_norm
            )
            self.maximum_residual_torque_nm = max(
                self.maximum_residual_torque_nm, torque_norm
            )
            if limit_reason is None and force_norm > self.force_limit_n:
                limit_reason = "WRIST_FT_RESULTANT_FORCE_ABORT"
            elif (
                limit_reason is None
                and torque_norm > effective_torque_limit_nm
            ):
                limit_reason = (
                    "WRIST_FT_PLANNED_CONTACT_RESULTANT_TORQUE_ABORT"
                    if planned_contact_mode
                    else "WRIST_FT_RESULTANT_TORQUE_ABORT"
                )
        else:
            planned_contact_mode = False
            effective_torque_limit_nm = self.torque_limit_nm
        if self.gate_enabled and limit_reason is not None:
            if self.first_ft_stop is None:
                self.first_ft_stop = {
                    "step": int(step),
                    "phase": str(phase),
                    "reason": limit_reason,
                    "gravity_compensated_task_wrench": (
                        None
                        if gravity_residual_task is None
                        else gravity_residual_task.tolist()
                    ),
                    "gravity_and_dynamic_compensated_task_wrench": (
                        None if residual_task is None else residual_task.tolist()
                    ),
                    "resultant_force_n": force_norm,
                    "resultant_torque_nm": torque_norm,
                }
            if self.stepper is None:
                raise RuntimeError("FT auditor is not attached to the stepper")
            if self.precontact_recovery_active:
                tolerance = 1.0e-6
                worsened = bool(
                    force_norm
                    > float(self.precontact_recovery_force_ceiling_n) + tolerance
                    or torque_norm
                    > float(self.precontact_recovery_torque_ceiling_nm) + tolerance
                )
                if worsened:
                    self.precontact_recovery_worsened = True
                    self.stepper.abort_reason = (
                        "WRIST_FT_PRECONTACT_RECOVERY_WRENCH_WORSENED_ABORT"
                    )
            else:
                self.stepper.abort_reason = limit_reason
        self.phase_counts[str(phase)] = self.phase_counts.get(str(phase), 0) + 1
        self.samples.append(
            {
                "step": int(step),
                "simulation_time_s": (int(step) + 1) * self.physics_dt_s,
                "phase": str(phase),
                "active_positions_rad": positions.tolist(),
                "active_velocities_rad_s": velocities.tolist(),
                "active_efforts_nm": efforts.tolist(),
                "active_targets_rad": targets.tolist(),
                "handbase_position_world_m": sensor[:3, 3].tolist(),
                "handbase_rotation_world_row_major": sensor[:3, :3].tolist(),
                "hand2arm_raw_wrench": raw_sensor.tolist(),
                "hand2arm_canonical_sensor_wrench": canonical_sensor.tolist(),
                "modeled_hand_gravity_sensor_wrench": gravity_sensor.tolist(),
                "run_specific_tare_canonical_sensor_wrench": (
                    None
                    if self.tare_canonical_sensor is None
                    else self.tare_canonical_sensor.tolist()
                ),
                "run_specific_tare_modeled_gravity_sensor_wrench": (
                    None
                    if self.tare_gravity_sensor is None
                    else self.tare_gravity_sensor.tolist()
                ),
                "gravity_difference_compensated_task_wrench": (
                    None
                    if gravity_residual_task is None
                    else gravity_residual_task.tolist()
                ),
                "dynamic_inertia_prediction": (
                    None
                    if dynamic_prediction is None
                    else {
                        "method": dynamic_prediction["method"],
                        "ready": dynamic_prediction["ready"],
                        "reason": dynamic_prediction["reason"],
                        "predicted_positive_inertia_wrench_sensor": (
                            None
                            if dynamic_prediction[
                                "predicted_positive_inertia_wrench_sensor"
                            ]
                            is None
                            else dynamic_prediction[
                                "predicted_positive_inertia_wrench_sensor"
                            ].tolist()
                        ),
                        "predicted_positive_inertia_wrench_task": (
                            None
                            if predicted_inertia_task is None
                            else predicted_inertia_task.tolist()
                        ),
                        "applied_to_safety_residual": dynamic_inertia_applied,
                        "force_balance": dynamic_prediction["force_balance"],
                        "physical_wrench_used_as_prediction_input": False,
                        "object_or_contact_truth_used": False,
                    }
                ),
                "gravity_and_dynamic_compensated_task_wrench": (
                    None if residual_task is None else residual_task.tolist()
                ),
                "resultant_force_n": force_norm,
                "resultant_torque_nm": torque_norm,
                "ft_safety_mode": (
                    "PLANNED_CONTACT"
                    if planned_contact_mode
                    else "STRICT_FREE_SPACE"
                ),
                "effective_force_limit_n": self.force_limit_n,
                "effective_torque_limit_nm": effective_torque_limit_nm,
                "ft_gate_enabled": bool(self.gate_enabled),
                "ft_limit_reason": limit_reason,
                "arm_control": {
                    "drive_target_rad": arm_control["drive_target_rad"],
                    "gravity_compensation_nm": arm_control[
                        "gravity_compensation_nm"
                    ],
                    "saturated": arm_control["saturated"],
                },
                "online_object_semantic_instance_or_contact_truth_used": False,
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "physical_source_joint": "hand2arm",
            "reaction_row": self.reaction_row,
            "canonical_from_raw": "NEGATIVE_IDENTITY_6",
            "tare_sample_count": len(self._tare_sensor_samples),
            "tare_canonical_sensor_wrench": (
                None
                if self.tare_canonical_sensor is None
                else self.tare_canonical_sensor.tolist()
            ),
            "tare_modeled_gravity_sensor_wrench": (
                None
                if self.tare_gravity_sensor is None
                else self.tare_gravity_sensor.tolist()
            ),
            "tare_minus_model_sensor_wrench": (
                None
                if self.tare_model_residual_sensor is None
                else self.tare_model_residual_sensor.tolist()
            ),
            "tare_model_consistency_checked_with_research_soft_limit": True,
            "gravity_difference_compensation": (
                "FROZEN_HAND_LINK_MASS_COM_MODEL_FROM_HAND_XACRO"
            ),
            "dynamic_inertia_compensation": {
                "enabled": self.dynamic_inertia_compensation_enabled,
                "enabled_phases": sorted(self.dynamic_inertia_enabled_phases),
                "algorithm": _CausalFrozenHandDynamicInertia.METHOD,
                "aggregate": (
                    None
                    if self.dynamic_inertia_aggregate is None
                    else _json_ready(self.dynamic_inertia_aggregate)
                ),
                "force_balance": (
                    "canonical=gravity+external_contact-inertia; "
                    "external_residual=gravity_residual+predicted_positive_inertia"
                ),
                "physical_wrench_used_as_prediction_input": False,
                "object_semantic_instance_or_contact_truth_used": False,
                "online_parameter_adaptation": False,
            },
            "maximum_residual_force_n": self.maximum_residual_force_n,
            "maximum_residual_torque_nm": self.maximum_residual_torque_nm,
            "force_limit_n": self.force_limit_n,
            "torque_limit_nm": self.torque_limit_nm,
            "planned_contact_torque_limit_nm": (
                self.planned_contact_torque_limit_nm
            ),
            "planned_contact_phase_selection": (
                "CONTROLLER_PHASE_ONLY_NO_CONTACT_OR_OBJECT_TRUTH"
            ),
            "first_safety_stop": self.first_ft_stop,
            "precontact_recovery": {
                "active": self.precontact_recovery_active,
                "force_ceiling_n": self.precontact_recovery_force_ceiling_n,
                "torque_ceiling_nm": self.precontact_recovery_torque_ceiling_nm,
                "worsened_abort": self.precontact_recovery_worsened,
                "normal_safety_limits_raised": False,
            },
            "phase_counts": dict(self.phase_counts),
            "online_contact_or_object_truth_used": False,
        }


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    return float(
        math.acos(np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0))
    )


def _capture_third_visual_servo_rgbd(
    *,
    repository: Path,
    arguments: argparse.Namespace,
    world: Any,
    stage: Any,
    simulation_app: Any,
    tabletop: Any,
    rgbd: Any,
    camera: Mapping[str, object],
    bindings: Mapping[str, Any],
    base_provider_input: Mapping[str, object],
    static_depth: np.ndarray,
    background_path: Path,
    plug_workspace: Any,
    plug_workspace_record: Mapping[str, object],
    expected_world_from_camera: np.ndarray,
    observed_intrinsics: np.ndarray,
    output: Path,
    reset_uuid: str,
    command_api_counter: Mapping[str, int],
    robot: Any,
    physics_step_index: int,
) -> dict[str, Any]:
    """Capture exactly one post-correction RGB-D frame without using truth."""

    from kcg_connector.isaac_d38999_rgbd_runtime import (
        capture_d38999_rgbd_raw_formal,
    )
    from kcg_connector.te_rgbd_observability import (
        evaluate_te_rgbd_observability,
    )
    from kcg_connector.te_rgbd_pose_provider import run_te_rgbd_pose_provider

    root = output / "third_observation"
    observation = root / "observation"
    capture_started = datetime.now(timezone.utc)
    command_counts_before = dict(command_api_counter)
    positions_before = _host_array(robot.get_dof_positions(indices=0))[0]
    velocities_before = _host_array(robot.get_dof_velocities(indices=0))[0]
    capture = capture_d38999_rgbd_raw_formal(
        bindings=bindings,
        simulation_app=simulation_app,
        world=world,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
        output_dir=observation,
        camera_clipping_range_m=tuple(camera["clipping_range_m"]),
        rt_subframes=int(
            arguments.same_reset_rgbd_document["timing"]["capture_rt_subframes"]
        ),
    )
    positions_after = _host_array(robot.get_dof_positions(indices=0))[0]
    velocities_after = _host_array(robot.get_dof_velocities(indices=0))[0]
    if capture.passed is not True or capture.rgb is None or capture.depth is None:
        failure = {
            "status": "THIRD_RGBD_CAPTURE_MISS",
            "ready": False,
            "raw_capture_metrics": capture.metrics,
            "robot_api_counts_unchanged": (
                dict(command_api_counter) == command_counts_before
            ),
            "truth_read": False,
        }
        _write_json_sealed(root / "failure.json", failure)
        return failure

    image_gates = arguments.same_reset_rgbd_document["image_gates"]
    observability = evaluate_te_rgbd_observability(
        rgb=capture.rgb,
        depth_m=capture.depth,
        static_depth_m=static_depth,
        intrinsics=observed_intrinsics,
        world_from_camera=expected_world_from_camera,
        workspaces=(plug_workspace,),
        minimum_key_width_px=4.0,
        minimum_foreground_depth_delta_m=float(
            image_gates["minimum_foreground_depth_delta_m"]
        ),
    )
    provider_input = copy.deepcopy(base_provider_input)
    capture_id = arguments.capture_id + "__post_correction"
    provider_input["capture_id"] = capture_id
    provider_input["capture_contract_sha256"] = file_sha256(
        arguments.visual_high_reobserve_config_path
    )
    provider_input["capture_time"] = {
        "clock_domain": "host_utc",
        "capture_started_at_utc": capture_started.isoformat(),
        "observed_frame_timestamp_utc": capture.metrics["capture_timestamp_utc"],
        "static_frame_timestamp_utc": base_provider_input["capture_time"][
            "static_frame_timestamp_utc"
        ],
    }
    provider_input["same_reset_evidence"] = {
        "reset_uuid": reset_uuid,
        "reset_count": 1,
        "resets_after_fresh_frame": 0,
        "robot_target_position_velocity_api_call_count": sum(
            int(value) for value in command_api_counter.values()
        ),
        "robot_motion_command_count": int(
            command_api_counter.get("set_dof_position_targets", 0)
        ),
        "settle_physics_steps_before_frame": int(physics_step_index),
        "provider_truth_read_before_result": False,
    }
    provider_input["ordinary_rgb"] = {
        "encoding": "png_rgb_uint8",
        "path_relative_to_manifest": "observation/rgb.png",
        "sha256": file_sha256(observation / "rgb.png"),
        "shape": list(capture.rgb.shape),
    }
    provider_input["ordinary_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "observation/depth_m.npy",
        "sha256": file_sha256(observation / "depth_m.npy"),
        "shape": list(capture.depth.shape),
    }
    background_dir = root / "background"
    background_dir.mkdir(parents=False, exist_ok=False)
    copied_background = background_dir / "depth_m.npy"
    shutil.copyfile(background_path, copied_background)
    if file_sha256(copied_background) != file_sha256(background_path):
        raise RuntimeError("third-observation background copy differs")
    provider_input["ordinary_static_scene_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "background/depth_m.npy",
        "sha256": file_sha256(copied_background),
        "shape": list(static_depth.shape),
        "frozen_source_path": base_provider_input[
            "ordinary_static_scene_depth"
        ]["frozen_source_path"],
        "frozen_source_sha256": base_provider_input[
            "ordinary_static_scene_depth"
        ]["frozen_source_sha256"],
    }
    provider_input["observability"] = observability
    provider_input["frozen_endpoint_workspaces_world_aabb_m"] = {
        "plug": plug_workspace_record
    }
    provider_input["control_authorized"] = False
    provider_input["pose_result"] = None
    provider_input_path = root / "provider_input.json"
    _write_json_sealed(provider_input_path, provider_input)
    try:
        provider_result = run_te_rgbd_pose_provider(
            provider_input_path, repository
        )
    except Exception as error:
        failure = {
            "status": "THIRD_PROVIDER_MISS",
            "ready": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "provider_input_sha256": file_sha256(provider_input_path),
            "robot_api_counts_unchanged": (
                dict(command_api_counter) == command_counts_before
            ),
            "truth_read": False,
        }
        _write_json_sealed(root / "failure.json", failure)
        return failure
    provider_result["provider_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    provider_result_path = root / "pose_provider_result.json"
    _write_json_sealed(provider_result_path, provider_result)
    sealed_at = datetime.now(timezone.utc)
    frame_time = datetime.fromisoformat(capture.metrics["capture_timestamp_utc"])
    frame_to_seal_s = (sealed_at - frame_time).total_seconds()
    ring = provider_result["transport_grasp_pose"]["derivation"].get(
        "rx180_axisymmetric_ring"
    )
    joint_delta = positions_after - positions_before
    readback_scale = np.float32(
        max(1.0, float(np.max(np.abs(positions_before))))
    )
    stationarity_tolerance = float(4.0 * np.spacing(readback_scale))
    captured_intrinsics = np.asarray(
        capture.metrics["camera"]["intrinsics"], dtype=np.float64
    )
    calibration_unchanged = bool(
        captured_intrinsics.shape == (3, 3)
        and np.allclose(
            captured_intrinsics,
            observed_intrinsics,
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    image_ready = bool(
        provider_result.get("provider_scope") == "TRANSPORT_PLUG_ONLY"
        and provider_result["transport_grasp_pose"].get("status")
        == "OBSERVED_AXIS_POSITION_YAW_FREE"
        and isinstance(ring, Mapping)
        and ring.get("fit_valid") is True
        and int(ring.get("point_count", 0))
        >= int(image_gates["minimum_rx180_ring_points"])
        and float(image_gates["minimum_ring_radius_m"])
        <= float(ring.get("radius_m", -1.0))
        <= float(image_gates["maximum_ring_radius_m"])
        and float(ring.get("absolute_radial_residual_p99_m", np.inf))
        <= float(image_gates["maximum_ring_absolute_residual_p99_m"])
        and frame_to_seal_s
        <= float(
            arguments.same_reset_rgbd_document["timing"][
                "maximum_frame_to_provider_seal_s"
            ]
        )
        and dict(command_api_counter) == command_counts_before
        and np.all(np.isfinite(joint_delta))
        and float(np.max(np.abs(joint_delta))) <= stationarity_tolerance
        and capture.metrics.get("timeline_pause", {}).get("paused_for_capture")
        is True
        and calibration_unchanged
    )
    report = {
        "schema_version": "kcg_te_visual_grasp_third_observation_v1",
        "capture_id": capture_id,
        "reset_uuid": reset_uuid,
        "ready": image_ready,
        "raw_capture_metrics": capture.metrics,
        "observability": observability,
        "provider_input_sha256": file_sha256(provider_input_path),
        "provider_result_sha256": file_sha256(provider_result_path),
        "provider_result_sealed_before_truth_read": True,
        "provider_sealed_at_utc": sealed_at.isoformat(),
        "frame_to_provider_seal_s": frame_to_seal_s,
        "physics_step_index": int(physics_step_index),
        "robot_api_counts_before_capture": command_counts_before,
        "robot_api_counts_after_provider": dict(command_api_counter),
        "joint_positions_before_capture_rad": positions_before.tolist(),
        "joint_positions_after_capture_rad": positions_after.tolist(),
        "joint_velocities_before_capture_rad_s": velocities_before.tolist(),
        "joint_velocities_after_capture_rad_s": velocities_after.tolist(),
        "joint_position_stationarity_tolerance_rad": stationarity_tolerance,
        "joint_position_stationarity_tolerance_method": (
            "FOUR_FLOAT32_SPACINGS_AT_MAX_ONE_OR_READBACK_MAGNITUDE"
        ),
        "captured_intrinsics_3x3": captured_intrinsics.tolist(),
        "frozen_intrinsics_unchanged": calibration_unchanged,
        "world_from_camera_source": "FROZEN_TE_NEAR_SIDE_CAMERA_CALIBRATION",
        "truth_read": False,
        "control_allowed_by_this_report": False,
    }
    report_path = root / "sealed_report.json"
    _write_json_sealed(report_path, report)
    return {
        "status": "READY" if image_ready else "THIRD_IMAGE_GATE_MISS",
        "ready": image_ready,
        "provider_input": provider_input,
        "provider_result": provider_result,
        "provider_input_path": provider_input_path,
        "provider_result_path": provider_result_path,
        "report_path": report_path,
        "frame_timestamp_utc": capture.metrics["capture_timestamp_utc"],
        "provider_sealed_at_utc": sealed_at.isoformat(),
        "joint_positions_after_capture_rad": positions_after,
        "joint_velocities_after_capture_rad_s": velocities_after,
    }


def _run_same_reset_rgbd_observe(
    repository,
    arguments,
    inputs,
    scene_entry,
    world,
    stage,
    scene,
    object_parts,
    grasp,
    ft_articulation,
    simulation_app,
    output,
    trace,
    *,
    visual_grasp_services=None,
):
    """Capture and seal fresh plug RGB-D before any robot target API call."""

    from PIL import Image
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.sensors.camera import Camera
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom, UsdLux

    from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene
    from kcg_connector.isaac_d38999_rgbd_runtime import (
        capture_d38999_rgbd_raw_formal,
    )
    from kcg_connector.rgbd_pose_bootstrap import RgbdCamera, load_rgbd_bootstrap
    from kcg_connector.te_rgbd_observability import (
        EndpointWorkspace,
        evaluate_te_rgbd_observability,
        world_from_camera_cv,
    )
    from kcg_connector.te_rgbd_pose_provider import run_te_rgbd_pose_provider

    document = arguments.same_reset_rgbd_document
    timing = document["timing"]
    image_gates = document["image_gates"]
    sources = arguments.same_reset_rgbd_sources
    reset_uuid = str(uuid.uuid4())
    trace["reset_uuid"] = reset_uuid
    trace["world_reset_count"] = 1
    trace["robot_target_position_velocity_api_call_count"] = 0
    trace["robot_motion_command_count"] = 0

    settle_steps = round(
        float(timing["settle_duration_s"]) / float(arguments.dynamic_settings["physics_dt_s"])
    )
    for _ in range(settle_steps):
        world.step(render=False)
    robot_read_view = Articulation(ARTICULATION_PATH)
    robot_positions = robot_read_view.get_dof_positions(indices=0).numpy()[0]
    if robot_positions.shape != (len(EXPECTED_DOF_NAMES),) or not np.all(
        np.isfinite(robot_positions)
    ):
        raise RuntimeError("read-only robot state is unavailable before RGB-D capture")
    trace["read_only_robot_joint_positions_after_settle_rad"] = (
        robot_positions.tolist()
    )
    trace["settle"] = {
        "physics_steps": settle_steps,
        "duration_s": float(timing["settle_duration_s"]),
        "robot_target_api_calls": 0,
        "object_truth_reads": 0,
    }

    tabletop_key = (
        "environment_scene_config"
        if scene_entry["scene_kind"] == "FREE_SINGLE_RIGID_ON_SHARED_FINITE_TABLE"
        else "scene_config"
    )
    tabletop = load_d38999_tabletop_scene(
        (repository / scene_entry[tabletop_key]).resolve()
    )
    rgbd = load_rgbd_bootstrap(sources["rgbd_bootstrap_config"])
    camera = arguments.same_reset_rgbd_camera_document["camera"]
    rgbd = replace(
        rgbd,
        camera=RgbdCamera(
            prim_path=str(camera["prim_path"]),
            frame_id=str(camera["frame_id"]),
            eye_m=tuple(float(value) for value in camera["eye_world_m"]),
            target_m=tuple(float(value) for value in camera["target_world_m"]),
            resolution=tuple(int(value) for value in camera["resolution_px"]),
            frequency_hz=int(camera["frequency_hz"]),
            warmup_frames=int(camera["warmup_frames"]),
        ),
    )
    observation_dir = output / "observation"
    bindings = {
        "Camera": Camera,
        "Gf": Gf,
        "Image": Image,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdLux": UsdLux,
        "rep": rep,
    }
    capture_started = datetime.now(timezone.utc)
    capture = capture_d38999_rgbd_raw_formal(
        bindings=bindings,
        simulation_app=simulation_app,
        world=world,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
        output_dir=observation_dir,
        camera_clipping_range_m=tuple(camera["clipping_range_m"]),
        rt_subframes=int(timing["capture_rt_subframes"]),
    )
    if capture.passed is not True or capture.rgb is None or capture.depth is None:
        failure = {
            "schema_version": "kcg_te_same_reset_rgbd_capture_failure_v1",
            "capture_id": arguments.capture_id,
            "reset_uuid": reset_uuid,
            "robot_motion_command_count": 0,
            "truth_read": False,
            "failure_reason": "RAW_RGBD_CAPTURE_FAILED",
            "raw_capture_metrics": capture.metrics,
        }
        _write_json_sealed(output / "capture_failure.json", failure)
        return 2

    background_dir = output / "background"
    background_dir.mkdir(parents=True, exist_ok=False)
    background_path = background_dir / "depth_m.npy"
    shutil.copyfile(sources["static_background_depth"], background_path)
    if file_sha256(background_path) != document["frozen_sources"][
        "static_background_depth_sha256"
    ]:
        raise RuntimeError("copied frozen static background identity differs")

    template = json.loads(
        sources["provider_input_template"].read_text(encoding="utf-8")
    )
    calibration = template["camera_calibration"]
    observed_intrinsics = np.asarray(
        capture.metrics["camera"]["intrinsics"], dtype=np.float64
    )
    if not np.allclose(
        observed_intrinsics,
        np.asarray(calibration["intrinsics_3x3"], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("fresh camera intrinsics differ from frozen calibration")
    expected_world_from_camera = world_from_camera_cv(
        camera["eye_world_m"], camera["target_world_m"]
    )
    if not np.allclose(
        expected_world_from_camera,
        np.asarray(
            calibration["world_from_camera_cv_row_major"], dtype=np.float64
        ).reshape(4, 4),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("near-side camera transform differs from frozen calibration")
    static_depth = np.asarray(
        np.load(background_path, allow_pickle=False), dtype=np.float64
    )
    plug_workspace_record = template[
        "frozen_endpoint_workspaces_world_aabb_m"
    ]["plug"]
    plug_workspace = EndpointWorkspace(
        name="plug",
        minimum_world_m=tuple(plug_workspace_record["minimum"]),
        maximum_world_m=tuple(plug_workspace_record["maximum"]),
        minimum_points=int(image_gates["minimum_plug_foreground_points"]),
        smallest_key_chord_m=0.0013208,
    )
    observability = evaluate_te_rgbd_observability(
        rgb=capture.rgb,
        depth_m=capture.depth,
        static_depth_m=static_depth,
        intrinsics=observed_intrinsics,
        world_from_camera=expected_world_from_camera,
        workspaces=(plug_workspace,),
        minimum_key_width_px=4.0,
        minimum_foreground_depth_delta_m=float(
            image_gates["minimum_foreground_depth_delta_m"]
        ),
    )

    provider_input = copy.deepcopy(template)
    provider_input["provider_scope"] = "TRANSPORT_PLUG_ONLY"
    provider_input["capture_id"] = arguments.capture_id
    provider_input["capture_contract_sha256"] = file_sha256(
        arguments.visual_high_reobserve_config_path
        if arguments.mode in VISION_MOTION_MODES
        else arguments.same_reset_rgbd_config_path
    )
    provider_input["capture_time"] = {
        "clock_domain": "host_utc",
        "capture_started_at_utc": capture_started.isoformat(),
        "observed_frame_timestamp_utc": capture.metrics[
            "capture_timestamp_utc"
        ],
        "static_frame_timestamp_utc": template["capture_time"][
            "static_frame_timestamp_utc"
        ],
    }
    provider_input["same_reset_evidence"] = {
        "reset_uuid": reset_uuid,
        "reset_count": 1,
        "resets_after_fresh_frame": 0,
        "robot_target_position_velocity_api_call_count": 0,
        "robot_motion_command_count": 0,
        "settle_physics_steps_before_frame": settle_steps,
        "provider_truth_read_before_result": False,
    }
    provider_input["ordinary_rgb"] = {
        "encoding": "png_rgb_uint8",
        "path_relative_to_manifest": "observation/rgb.png",
        "sha256": file_sha256(observation_dir / "rgb.png"),
        "shape": list(capture.rgb.shape),
    }
    provider_input["ordinary_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "observation/depth_m.npy",
        "sha256": file_sha256(observation_dir / "depth_m.npy"),
        "shape": list(capture.depth.shape),
    }
    provider_input["ordinary_static_scene_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "background/depth_m.npy",
        "sha256": file_sha256(background_path),
        "shape": list(static_depth.shape),
        "frozen_source_path": str(
            sources["static_background_depth"].relative_to(repository)
        ),
        "frozen_source_sha256": file_sha256(
            sources["static_background_depth"]
        ),
    }
    provider_input["observability"] = observability
    provider_input["frozen_endpoint_workspaces_world_aabb_m"] = {
        "plug": plug_workspace_record
    }
    provider_input["te_cad_models"] = {
        "plug": template["te_cad_models"]["plug"]
    }
    provider_input["formal_gate"] = {
        "required_outputs": ["transport_grasp_pose"],
        "key_observation_required": False,
        "miss_authorizes_motion": False,
        "pose_result_from_scene_generation_truth_forbidden": True,
        "missing_endpoint_result": "MISS_ENDPOINT_NOT_FOUND",
        "ambiguous_key_result": "NOT_APPLICABLE_TO_TRANSPORT_PLUG_ONLY",
        "unobserved_key_result": "NOT_APPLICABLE_TO_TRANSPORT_PLUG_ONLY",
    }
    provider_input["provider_callable_contract"] = {
        "implementation_status": "TE_TRANSPORT_PLUG_ONLY_SAME_RESET_V1",
        "input": "fresh ordinary RGB/depth plus frozen static depth/calibration/CAD",
        "output": "yaw-free transport plug pose or fail-closed MISS",
    }
    provider_input["control_authorized"] = False
    provider_input["pose_result"] = None
    provider_input_path = output / "provider_input.json"
    _write_json_sealed(provider_input_path, provider_input)

    try:
        provider_result = run_te_rgbd_pose_provider(
            provider_input_path, repository
        )
    except Exception as error:
        _write_json_sealed(
            output / "capture_failure.json",
            {
                "schema_version": "kcg_te_same_reset_rgbd_capture_failure_v1",
                "capture_id": arguments.capture_id,
                "reset_uuid": reset_uuid,
                "robot_motion_command_count": 0,
                "truth_read": False,
                "failure_reason": "PLUG_FOREGROUND_OR_PROVIDER_FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "provider_input_sha256": file_sha256(provider_input_path),
            },
        )
        return 2
    provider_generated_at = datetime.now(timezone.utc)
    provider_result["provider_generated_at_utc"] = provider_generated_at.isoformat()
    provider_result_path = output / "pose_provider_result.json"
    _write_json_sealed(provider_result_path, provider_result)
    provider_sealed_at = datetime.now(timezone.utc)
    frame_time = datetime.fromisoformat(capture.metrics["capture_timestamp_utc"])
    frame_to_seal_s = (provider_sealed_at - frame_time).total_seconds()
    ring = provider_result["transport_grasp_pose"]["derivation"].get(
        "rx180_axisymmetric_ring"
    )
    image_pass = bool(
        provider_result.get("provider_scope") == "TRANSPORT_PLUG_ONLY"
        and provider_result["transport_grasp_pose"].get("status")
        == "OBSERVED_AXIS_POSITION_YAW_FREE"
        and isinstance(ring, Mapping)
        and ring.get("fit_valid") is True
        and int(ring.get("point_count", 0))
        >= int(image_gates["minimum_rx180_ring_points"])
        and float(image_gates["minimum_ring_radius_m"])
        <= float(ring.get("radius_m", -1.0))
        <= float(image_gates["maximum_ring_radius_m"])
        and float(ring.get("absolute_radial_residual_p99_m", np.inf))
        <= float(image_gates["maximum_ring_absolute_residual_p99_m"])
        and frame_to_seal_s
        <= float(timing["maximum_frame_to_provider_seal_s"])
        and provider_result["transport_grasp_pose"].get(
            "transport_grasp_control_allowed"
        )
        is False
        and provider_result.get("robot_command_count") == 0
    )
    raw_report = {
        "schema_version": "kcg_te_same_reset_rgbd_sealed_capture_v1",
        "capture_id": arguments.capture_id,
        "reset_uuid": reset_uuid,
        "world_reset_count": 1,
        "resets_after_fresh_frame": 0,
        "robot_target_position_velocity_api_call_count": 0,
        "robot_motion_command_count": 0,
        "raw_capture_metrics": capture.metrics,
        "observability": observability,
        "provider_input_sha256": file_sha256(provider_input_path),
        "provider_result_sha256": file_sha256(provider_result_path),
        "provider_result_sealed_before_truth_read": True,
        "provider_sealed_at_utc": provider_sealed_at.isoformat(),
        "frame_to_provider_seal_s": frame_to_seal_s,
        "frozen_background_sha256": file_sha256(background_path),
        "image_and_freshness_gate_pass": image_pass,
        "truth_read": False,
        "control_authorized": False,
    }
    _write_json_sealed(output / "sealed_capture_report.json", raw_report)
    if not image_pass:
        _write_json_sealed(
            output / "capture_failure.json",
            {
                "schema_version": "kcg_te_same_reset_rgbd_capture_failure_v1",
                "capture_id": arguments.capture_id,
                "reset_uuid": reset_uuid,
                "robot_motion_command_count": 0,
                "truth_read": False,
                "failure_reason": "FOREGROUND_RING_OR_FRESHNESS_GATE_FAILED",
                "sealed_capture_report_sha256": file_sha256(
                    output / "sealed_capture_report.json"
                ),
            },
        )
        return 2

    if arguments.mode in VISION_MOTION_MODES:
        if ft_articulation is None:
            raise RuntimeError("visual high mode lacks the physical FT reader")
        return _run_visual_high_reobserve_after_first_provider(
            repository=repository,
            arguments=arguments,
            inputs=inputs,
            grasp=grasp,
            world=world,
            stage=stage,
            scene=scene,
            object_parts=object_parts,
            ft_articulation=ft_articulation,
            simulation_app=simulation_app,
            output=output,
            trace=trace,
            provider_input=provider_input,
            provider_result=provider_result,
            provider_result_path=provider_result_path,
            reset_uuid=reset_uuid,
            tabletop=tabletop,
            rgbd=rgbd,
            camera=camera,
            bindings=bindings,
            static_depth=static_depth,
            background_path=background_path,
            plug_workspace=plug_workspace,
            plug_workspace_record=plug_workspace_record,
            expected_world_from_camera=expected_world_from_camera,
            observed_intrinsics=observed_intrinsics,
            visual_grasp_services=visual_grasp_services,
        )

    def truth_pose():
        position, quaternion = object_parts[0].get_world_pose()
        if hasattr(position, "detach"):
            position = position.detach().cpu()
        if hasattr(quaternion, "detach"):
            quaternion = quaternion.detach().cpu()
        return (
            np.asarray(position, dtype=np.float64),
            np.asarray(quaternion, dtype=np.float64),
        )

    start_position, start_quaternion = truth_pose()
    start_axis = _quaternion_wxyz_rotation(start_quaternion) @ np.asarray(
        (0.0, 0.0, 1.0), dtype=np.float64
    )
    estimated = provider_result["transport_grasp_pose"]
    estimated_position = np.asarray(estimated["position_xyz_m"], dtype=np.float64)
    estimated_axis = np.asarray(estimated["outward_axis_world"], dtype=np.float64)
    axis_tilt = float(
        np.arccos(
            np.clip(
                float(start_axis @ estimated_axis)
                / float(np.linalg.norm(start_axis) * np.linalg.norm(estimated_axis)),
                -1.0,
                1.0,
            )
        )
    )
    z_error = float(estimated_position[2] - start_position[2])
    tail_steps = round(
        float(timing["tail_observation_duration_s"])
        / float(arguments.dynamic_settings["physics_dt_s"])
    )
    for _ in range(tail_steps):
        world.step(render=False)
    end_position, end_quaternion = truth_pose()
    tail_translation = float(np.linalg.norm(end_position - start_position))
    posthoc_gates = document["posthoc_research_gates"]
    posthoc_pass = bool(
        abs(z_error)
        <= float(posthoc_gates["maximum_absolute_support_z_error_m"])
        and axis_tilt <= float(posthoc_gates["maximum_axis_tilt_rad"])
        and tail_translation
        <= float(posthoc_gates["maximum_tail_translation_m"])
    )
    posthoc = {
        "schema_version": "kcg_te_same_reset_rgbd_posthoc_truth_v1",
        "capture_id": arguments.capture_id,
        "reset_uuid": reset_uuid,
        "provider_result_sha256": file_sha256(provider_result_path),
        "provider_result_sealed_before_truth_read": True,
        "truth_read_count": 2,
        "truth_returned_to_provider_or_controller": False,
        "robot_target_position_velocity_api_call_count": 0,
        "robot_motion_command_count": 0,
        "estimated_position_xyz_m": estimated_position.tolist(),
        "posthoc_truth_start_position_xyz_m": start_position.tolist(),
        "signed_support_z_error_m": z_error,
        "estimated_outward_axis_world": estimated_axis.tolist(),
        "posthoc_truth_outward_axis_world": start_axis.tolist(),
        "axis_tilt_error_rad": axis_tilt,
        "tail_duration_s": float(timing["tail_observation_duration_s"]),
        "tail_steps": tail_steps,
        "posthoc_truth_end_position_xyz_m": end_position.tolist(),
        "posthoc_truth_start_orientation_wxyz": start_quaternion.tolist(),
        "posthoc_truth_end_orientation_wxyz": end_quaternion.tolist(),
        "tail_translation_m": tail_translation,
        "preregistered_research_gates": posthoc_gates,
        "posthoc_research_gate_pass": posthoc_pass,
        "control_authorized": False,
        "evidence_limit": (
            "ONE_SIMULATION_ONLY_SAME_RESET_SAMPLE_NOT_3SIGMA_NOT_HARDWARE_"
            "OR_MANUFACTURING_CERTIFICATION"
        ),
    }
    _write_json_sealed(output / "posthoc_truth_evaluation.json", posthoc)
    trace.update(
        {
            "provider_result_sha256": file_sha256(provider_result_path),
            "provider_result_sealed_before_truth_read": True,
            "posthoc_truth_evaluation_sha256": file_sha256(
                output / "posthoc_truth_evaluation.json"
            ),
            "robot_target_position_velocity_api_call_count": 0,
            "robot_motion_command_count": 0,
            "formal_dynamic_pass": False,
            "control_authorized": False,
        }
    )
    _write_json_sealed(output / "trace.json", trace)
    print(json.dumps(posthoc, ensure_ascii=False, indent=2))
    return 0 if posthoc_pass else 2


def _audit_visual_second_short_prefix(
    *,
    repository: Path,
    inputs: Any,
    first_planned_high_arm: Sequence[float],
    actual_start_arm: Sequence[float],
    actual_start_hand: Sequence[float],
    second_target_arm: Sequence[float],
    frozen_hand_positions: Sequence[float],
    second_pregrasp_world_from_hand: np.ndarray,
    approach_direction_world: Sequence[float],
    high_panel: Mapping[str, object],
    prefix_contract: Mapping[str, object],
    gate_contract: Mapping[str, object],
    world_from_transport_object: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite, truth-free clearance audit for one visual short-path prefix."""

    import fcl

    from kcg_connector.grasp.robust.collision_roster import (
        load_authoritative_collision_link_roster,
    )
    from kcg_connector.grasp.robust.hand_model import rpy_rotation
    from kcg_connector.grasp.robust.object_model import load_stl_mesh

    first_high = np.asarray(first_planned_high_arm, dtype=np.float64)
    actual_start = np.asarray(actual_start_arm, dtype=np.float64)
    actual_hand = np.asarray(actual_start_hand, dtype=np.float64)
    target_arm = np.asarray(second_target_arm, dtype=np.float64)
    hand = np.asarray(frozen_hand_positions, dtype=np.float64)
    approach = np.asarray(approach_direction_world, dtype=np.float64)
    pregrasp = np.asarray(second_pregrasp_world_from_hand, dtype=np.float64)
    if not (
        first_high.shape == actual_start.shape == target_arm.shape == (7,)
        and actual_hand.shape == hand.shape == (4,)
        and approach.shape == (3,)
        and pregrasp.shape == (4, 4)
        and all(
            np.all(np.isfinite(row))
            for row in (
                first_high,
                actual_start,
                actual_hand,
                target_arm,
                hand,
                approach,
                pregrasp,
            )
        )
        and math.isclose(
            float(np.linalg.norm(approach)), 1.0, rel_tol=0.0, abs_tol=1.0e-9
        )
    ):
        raise ValueError("second visual short-prefix input is invalid")

    target_fk = inputs.robot_model.forward_kinematics(
        tuple(np.concatenate((target_arm, hand))), enforce_limits=False
    )
    target_hand = np.asarray(target_fk["handbase_link"], dtype=np.float64)
    target_delta = target_hand[:3, 3] - pregrasp[:3, 3]
    target_clearance = -float(target_delta @ approach)
    target_lateral_error = float(
        np.linalg.norm(target_delta + target_clearance * approach)
    )
    target_rotation_error = _rotation_error_rad(
        target_hand[:3, :3], pregrasp[:3, :3]
    )
    target_clearance_error = abs(
        target_clearance
        - float(prefix_contract["target_clearance_from_pregrasp_m"])
    )

    roster_path = str(gate_contract["collision_roster"])
    configured_roster = str(inputs.config.section("inputs")["collision_roster"])
    if roster_path != configured_roster:
        raise ValueError("second-prefix collision roster differs from grasp input")
    roster = load_authoritative_collision_link_roster(
        roster_path, repository_root=repository
    )
    local_vertices = {
        name: np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
        for name, triangles in inputs.hand_collision_triangles_by_link.items()
    }
    for link in roster.links:
        if link.link_name in local_vertices:
            continue
        mesh, provenance = load_stl_mesh(
            link.absolute_path, unit=link.unit, orient_outward=False
        )
        if provenance.source_sha256 != link.sha256:
            raise ValueError(
                f"second-prefix collision mesh changed for {link.link_name}"
            )
        triangles = np.asarray(mesh.face_vertices_m, dtype=np.float64)
        triangles = triangles * np.asarray(link.scale, dtype=np.float64)
        triangles = (
            triangles @ rpy_rotation(link.origin_rpy_rad).T
            + np.asarray(link.origin_xyz_m, dtype=np.float64)
        )
        local_vertices[link.link_name] = triangles.reshape(-1, 3)
    expected_links = {link.link_name for link in roster.links}
    if (
        set(local_vertices) != expected_links
        or len(local_vertices) != int(gate_contract["collision_link_count"])
        or any(
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.all(np.isfinite(vertices))
            for vertices in local_vertices.values()
        )
    ):
        raise ValueError("second-prefix complete collision roster is unavailable")

    def state_geometry(
        arm_positions: np.ndarray,
        hand_positions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], Mapping[str, np.ndarray]]:
        transforms = inputs.robot_model.forward_kinematics(
            tuple(np.concatenate((arm_positions, hand_positions))),
            enforce_limits=False,
        )
        world_vertices: dict[str, np.ndarray] = {}
        for link_name, vertices in local_vertices.items():
            transform = np.asarray(transforms[link_name], dtype=np.float64)
            world_vertices[link_name] = (
                vertices @ transform[:3, :3].T + transform[:3, 3]
            )
        return world_vertices, transforms

    baseline_vertices, _ = state_geometry(first_high, hand)
    state_count = int(gate_contract["state_count"])
    motion_command_count = int(prefix_contract["motion_command_count"])
    if (
        state_count != int(prefix_contract["runtime_state_count"])
        or state_count != motion_command_count + 1
        or motion_command_count < 1
    ):
        raise ValueError("second-prefix runtime state count differs")
    states: list[tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]] = []
    maximum_link_displacement = {name: 0.0 for name in local_vertices}
    state_targets = [(actual_start, actual_hand)]
    for command_index in range(motion_command_count):
        blend = control.minimum_jerk_blend(
            (command_index + 1) / motion_command_count
        )
        state_targets.append(
            ((1.0 - blend) * actual_start + blend * target_arm, hand)
        )
    for arm, hand_state in state_targets:
        vertices, transforms = state_geometry(arm, hand_state)
        states.append((arm, hand_state, transforms))
        for link_name in local_vertices:
            maximum_link_displacement[link_name] = max(
                maximum_link_displacement[link_name],
                float(
                    np.max(
                        np.linalg.norm(
                            vertices[link_name] - baseline_vertices[link_name], axis=1
                        )
                    )
                ),
            )
    maximum_displacement = max(maximum_link_displacement.values())

    def fcl_model(vertices: np.ndarray) -> Any:
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        model = fcl.BVHModel()
        model.beginModel(len(vertices), len(faces))
        model.addSubModel(np.ascontiguousarray(vertices), faces)
        model.endModel()
        return model

    collision_objects = {
        name: fcl.CollisionObject(fcl_model(vertices))
        for name, vertices in local_vertices.items()
    }
    obstacle_rule = str(gate_contract["obstacle_lower_bound_rule"])
    exact_obstacle_rule = (
        "EXACT_TABLE_AND_YAW_SWEPT_CAD_CYLINDER_FCL_MINUS_DERIVED_POSE_"
        "ENVELOPE"
    )
    exact_obstacles = obstacle_rule == exact_obstacle_rule
    object_obstacle = None
    table_obstacle = None
    minimum_nominal_object = math.inf
    minimum_nominal_object_link: str | None = None
    minimum_nominal_object_state: int | None = None
    minimum_table = math.inf
    minimum_table_link: str | None = None
    minimum_table_state: int | None = None
    first_obstacle_collision: list[object] | None = None
    if exact_obstacles:
        world_from_object = np.asarray(
            world_from_transport_object, dtype=np.float64
        )
        if world_from_object.shape != (4, 4) or not np.all(
            np.isfinite(world_from_object)
        ):
            raise ValueError("exact short-prefix object transform is invalid")
        object_vertices = np.asarray(
            inputs.object_contract.model.mesh.face_vertices_m,
            dtype=np.float64,
        ).reshape(-1, 3)
        cylinder_contract = gate_contract["yaw_swept_cad_cylinder"]
        cylinder_radius = float(
            np.max(np.linalg.norm(object_vertices[:, :2], axis=1))
        )
        cylinder_z_min = float(np.min(object_vertices[:, 2]))
        cylinder_z_max = float(np.max(object_vertices[:, 2]))
        cylinder_z_center = 0.5 * (cylinder_z_min + cylinder_z_max)
        cylinder_length = cylinder_z_max - cylinder_z_min
        cylinder_origin_radius = math.hypot(
            cylinder_radius,
            max(abs(cylinder_z_min), abs(cylinder_z_max)),
        )
        if not (
            inputs.object_contract.model.provenance.source_sha256
            == cylinder_contract["registration_cad"]["sha256"]
            and math.isclose(
                cylinder_radius,
                float(cylinder_contract["radius_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                cylinder_z_min,
                float(cylinder_contract["z_min_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                cylinder_z_max,
                float(cylinder_contract["z_max_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                cylinder_origin_radius,
                float(cylinder_contract["origin_radius_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("yaw-swept CAD cylinder geometry differs")
        world_from_cylinder = world_from_object.copy()
        world_from_cylinder[:3, 3] = (
            world_from_object[:3, :3]
            @ np.asarray((0.0, 0.0, cylinder_z_center), dtype=np.float64)
            + world_from_object[:3, 3]
        )
        object_obstacle = fcl.CollisionObject(
            fcl.Cylinder(cylinder_radius, cylinder_length),
            fcl.Transform(
                world_from_cylinder[:3, :3], world_from_cylinder[:3, 3]
            ),
        )
        bounds = np.asarray(inputs.table_xy_bounds_m, dtype=np.float64)
        table_size_x = float(bounds[0, 1] - bounds[0, 0])
        table_size_y = float(bounds[1, 1] - bounds[1, 0])
        table_center = np.asarray(
            (
                float(np.mean(bounds[0])),
                float(np.mean(bounds[1])),
                float(inputs.table_top_z_m - 0.5),
            ),
            dtype=np.float64,
        )
        table_obstacle = fcl.CollisionObject(
            fcl.Box(table_size_x, table_size_y, 1.0),
            fcl.Transform(table_center),
        )
    adjacent_pairs = {
        tuple(sorted((joint.parent_link, joint.child_link)))
        for joint in inputs.robot_model.joints.values()
        if joint.parent_link in collision_objects
        and joint.child_link in collision_objects
    }
    link_names = tuple(sorted(collision_objects))
    checked_pairs = tuple(
        (first, second)
        for index, first in enumerate(link_names)
        for second in link_names[index + 1 :]
        if tuple(sorted((first, second))) not in adjacent_pairs
    )
    collision_request = fcl.CollisionRequest(
        num_max_contacts=1, enable_contact=False
    )
    distance_request = fcl.DistanceRequest(enable_nearest_points=False)
    minimum_self_clearance = math.inf
    limiting_self_pair: list[object] | None = None
    first_self_collision: list[object] | None = None
    for state_index, (_, _, transforms) in enumerate(states):
        for link_name, collision_object in collision_objects.items():
            transform = np.asarray(transforms[link_name], dtype=np.float64)
            collision_object.setTransform(
                fcl.Transform(transform[:3, :3], transform[:3, 3])
            )
        if exact_obstacles:
            assert object_obstacle is not None and table_obstacle is not None
            for link_name, collision_object in collision_objects.items():
                for obstacle_name, obstacle in (
                    ("yaw_swept_cad_cylinder", object_obstacle),
                    ("table", table_obstacle),
                ):
                    collided = bool(
                        fcl.collide(
                            collision_object,
                            obstacle,
                            collision_request,
                            fcl.CollisionResult(),
                        )
                    )
                    distance = (
                        0.0
                        if collided
                        else float(
                            fcl.distance(
                                collision_object,
                                obstacle,
                                distance_request,
                                fcl.DistanceResult(),
                            )
                        )
                    )
                    if not math.isfinite(distance) or distance < 0.0:
                        raise RuntimeError(
                            "second-prefix FCL obstacle distance is unavailable"
                        )
                    if collided and first_obstacle_collision is None:
                        first_obstacle_collision = [
                            state_index,
                            link_name,
                            obstacle_name,
                        ]
                    if (
                        obstacle_name == "yaw_swept_cad_cylinder"
                        and distance < minimum_nominal_object
                    ):
                        minimum_nominal_object = distance
                        minimum_nominal_object_link = link_name
                        minimum_nominal_object_state = state_index
                    if obstacle_name == "table" and distance < minimum_table:
                        minimum_table = distance
                        minimum_table_link = link_name
                        minimum_table_state = state_index
        for first, second in checked_pairs:
            if fcl.collide(
                collision_objects[first],
                collision_objects[second],
                collision_request,
                fcl.CollisionResult(),
            ):
                first_self_collision = [state_index, first, second]
                minimum_self_clearance = 0.0
                break
            distance = float(
                fcl.distance(
                    collision_objects[first],
                    collision_objects[second],
                    distance_request,
                    fcl.DistanceResult(),
                )
            )
            if not math.isfinite(distance) or distance < 0.0:
                raise RuntimeError("second-prefix FCL self-distance is unavailable")
            if distance < minimum_self_clearance:
                minimum_self_clearance = distance
                limiting_self_pair = [first, second, state_index]
        if first_self_collision is not None:
            break

    lower, upper = inputs.robot_model.joint_limit_vectors()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    minimum_joint_margin = min(
        float(
            np.min(
                np.minimum(
                    active - lower,
                    upper - active,
                )
            )
        )
        for arm, hand_state, _ in states
        for active in (np.concatenate((arm, hand_state)),)
    )
    exact_obstacle_diagnostics = None
    if exact_obstacles:
        object_envelope = float(
            gate_contract["pose_envelope_derivation"][
                "derived_surface_displacement_envelope_m"
            ]
        )
        obstacle_lower_bounds = {
            "table": minimum_table,
            "range_robust_plug": minimum_nominal_object - object_envelope,
            "fixture": float(high_panel["minimum_fixture_clearance_m"])
            - maximum_displacement,
        }
        exact_obstacle_diagnostics = {
            "object_obstacle_kind": "YAW_SWEPT_CAD_BOUNDING_CYLINDER",
            "yaw_swept_cylinder_contract": gate_contract[
                "yaw_swept_cad_cylinder"
            ],
            "minimum_yaw_swept_cylinder_distance_m": minimum_nominal_object,
            "minimum_yaw_swept_cylinder_link": minimum_nominal_object_link,
            "minimum_yaw_swept_cylinder_state_index": (
                minimum_nominal_object_state
            ),
            "derived_pose_surface_displacement_envelope_m": object_envelope,
            "minimum_table_distance_m": minimum_table,
            "minimum_table_link": minimum_table_link,
            "minimum_table_state_index": minimum_table_state,
            "first_obstacle_collision": first_obstacle_collision,
        }
    else:
        obstacle_lower_bounds = {
            "table": float(high_panel["minimum_table_clearance_m"])
            - maximum_displacement,
            "range_robust_plug": float(
                high_panel["minimum_range_robust_plug_clearance_m"]
            )
            - maximum_displacement,
            "fixture": float(high_panel["minimum_fixture_clearance_m"])
            - maximum_displacement,
        }
    obstacle_gate = float(
        gate_contract["obstacle_clearance_lower_bound_minimum_m"]
    )
    self_gate = float(gate_contract["nonadjacent_self_clearance_minimum_m"])
    target_geometry_pass = bool(
        target_clearance_error <= 1.0e-5
        and target_lateral_error <= 1.0e-5
        and target_rotation_error <= 1.0e-5
    )
    gate_pass = bool(
        target_geometry_pass
        and all(value >= obstacle_gate for value in obstacle_lower_bounds.values())
        and first_obstacle_collision is None
        and first_self_collision is None
        and minimum_self_clearance >= self_gate
        and minimum_joint_margin > 0.0
    )
    return {
        "schema_version": "kcg_te_visual_second_short_prefix_static_audit_v1",
        "method": gate_contract["method"],
        "state_count": state_count,
        "motion_command_target_state_count": motion_command_count,
        "settle_target_already_included_as_final_motion_target": True,
        "collision_link_count": len(local_vertices),
        "nonadjacent_pair_count": len(checked_pairs),
        "trajectory_start": prefix_contract["trajectory_start"],
        "actual_start_hand_minus_frozen_rad": (actual_hand - hand).tolist(),
        "surface_displacement_reference": prefix_contract[
            "surface_displacement_reference"
        ],
        "target_method": prefix_contract.get(
            "target_method", "PLANNER_WAYPOINT_INDEX"
        ),
        "target_waypoint_index": (
            None
            if "target_waypoint_index" not in prefix_contract
            else int(prefix_contract["target_waypoint_index"])
        ),
        "target_clearance_from_pregrasp_m": target_clearance,
        "target_clearance_error_m": target_clearance_error,
        "target_lateral_error_m": target_lateral_error,
        "target_rotation_error_rad": target_rotation_error,
        "target_geometry_pass": target_geometry_pass,
        "maximum_link_surface_displacement_from_first_planned_high_m": (
            maximum_displacement
        ),
        "limiting_displacement_link": max(
            maximum_link_displacement, key=maximum_link_displacement.get
        ),
        "maximum_link_surface_displacement_m": maximum_link_displacement,
        "conservative_obstacle_clearance_lower_bounds_m": obstacle_lower_bounds,
        "obstacle_distance_method": obstacle_rule,
        "exact_obstacle_diagnostics": exact_obstacle_diagnostics,
        "minimum_required_obstacle_clearance_m": obstacle_gate,
        "minimum_nonadjacent_self_clearance_m": minimum_self_clearance,
        "minimum_required_nonadjacent_self_clearance_m": self_gate,
        "limiting_self_pair": limiting_self_pair,
        "first_self_collision": first_self_collision,
        "minimum_active_joint_limit_margin_rad": minimum_joint_margin,
        "pass": gate_pass,
        "online_object_semantic_instance_or_contact_truth_used": False,
        "continuous_joint_space_proof_claimed": False,
        "evidence_limit": (
            "FINITE_STATE_SIMULATION_GEOMETRY_GATE_NOT_CONTINUOUS_PATH_PROOF"
        ),
    }


def _run_visual_high_reobserve_after_first_provider(
    *,
    repository: Path,
    arguments: argparse.Namespace,
    inputs: Any,
    grasp: Mapping[str, object],
    world: Any,
    stage: Any,
    scene: Mapping[str, object],
    object_parts: Sequence[Any],
    ft_articulation: Any,
    simulation_app: Any,
    output: Path,
    trace: dict[str, Any],
    provider_input: Mapping[str, object],
    provider_result: Mapping[str, object],
    provider_result_path: Path,
    reset_uuid: str,
    tabletop: Any,
    rgbd: Any,
    camera: Mapping[str, object],
    bindings: Mapping[str, Any],
    static_depth: np.ndarray,
    background_path: Path,
    plug_workspace: Any,
    plug_workspace_record: Mapping[str, object],
    expected_world_from_camera: np.ndarray,
    observed_intrinsics: np.ndarray,
    visual_grasp_services: Mapping[str, object] | None = None,
) -> int | tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Run the sealed visual approach, optionally handing off to grasp/lift."""

    from kcg_connector.isaac_d38999_rgbd_runtime import (
        capture_d38999_rgbd_raw_formal,
    )
    from kcg_connector.te_rgbd_observability import (
        evaluate_te_rgbd_observability,
    )
    from kcg_connector.te_rgbd_pose_provider import run_te_rgbd_pose_provider

    high = arguments.visual_high_reobserve_document
    if high is None:
        raise RuntimeError("visual high-reobserve contract was not loaded")
    grasp_servo = arguments.mode == VISION_GRASP_SERVO_MODE
    if grasp_servo:
        if visual_grasp_services is None:
            raise RuntimeError("visual grasp services are unavailable")
        friction = scene["contact_friction_material_audit"]
        pair = friction["contact_pair"]
        if not (
            friction.get("applied") is True
            and math.isclose(
                float(pair["effective_static_friction"]),
                0.45,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            and math.isclose(
                float(pair["effective_dynamic_friction"]),
                0.45,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise RuntimeError("visual grasp did not read back effective friction 0.45")
        transport_relation_path = (
            repository
            / "src/kcg_connector/config/te_transport_grasp_relation_v2.yaml"
        ).resolve()
        transport_relation, _ = load_transport_grasp_relation(
            transport_relation_path, repository
        )
        if (
            transport_relation.get("relation_id")
            != "te_development_transport_grasp_v2"
            or float(
                transport_relation["hand_contract"][
                    "contact_confirmation_velocity_absolute_max_rad_s"
                ]
            )
            != 0.14
        ):
            raise RuntimeError("corrected visual transport relation differs")
        visual_control_plan = dict(grasp["control_plan"])
        visual_control_plan["object_from_hand_row_major"] = list(
            transport_relation["transform"][
                "object_from_hand_base_row_major"
            ]
        )
        visual_grasp = dict(grasp)
        visual_grasp["control_plan"] = visual_control_plan
        visual_payload_model = dict(visual_grasp_services["payload_model"])
        center_delta = (
            np.zeros(3, dtype=np.float64)
            if scene["requested_center_of_mass_delta_object_m"] is None
            else np.asarray(
                scene["requested_center_of_mass_delta_object_m"],
                dtype=np.float64,
            )
        )
        center_object = (
            np.asarray(
                inputs.object_contract.model.center_of_mass_m,
                dtype=np.float64,
            )
            + center_delta
        )
        object_from_hand = np.asarray(
            visual_control_plan["object_from_hand_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        center_hand = (
            np.linalg.inv(object_from_hand)
            @ np.concatenate((center_object, (1.0,)))
        )[:3]
        visual_payload_model["center_of_mass_from_hand_m"] = center_hand.tolist()
        visual_payload_model["visual_relation_id"] = transport_relation[
            "relation_id"
        ]
        trace["visual_grasp_user_authorization"] = {
            "simulation_only": True,
            "same_reset_visual_servo_grasp_lift": True,
            "source_high_contract_used_only_for_camera_ft_and_high_motion": True,
            "assembly_control_authorized": False,
            "formal_postgrasp_main_key_observed": False,
        }
    else:
        transport_relation_path = (
            repository / high["frozen_grasp_relation"]["path"]
        ).resolve()
        visual_control_plan = grasp["control_plan"]
        visual_grasp = grasp
        visual_payload_model = None
    high_config_path = arguments.visual_high_reobserve_config_path
    try:
        output.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            "fresh visual target evidence must stay inside the repository"
        ) from error

    # The 126-state collision panel is centered on the independent same-reset
    # provider.  A new provider may only move inside its preregistered finite
    # research set; no object truth is read for this comparison.
    reference_path = _bound_repository_file(
        repository,
        high["source_evidence"]["same_reset_provider"],
        "high-mode reference provider",
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    observed_transport = provider_result["transport_grasp_pose"]
    reference_transport = reference["transport_grasp_pose"]
    observed_position = np.asarray(
        observed_transport["position_xyz_m"], dtype=np.float64
    )
    reference_position = np.asarray(
        reference_transport["position_xyz_m"], dtype=np.float64
    )
    observed_axis = np.asarray(
        observed_transport["outward_axis_world"], dtype=np.float64
    )
    reference_axis = np.asarray(
        reference_transport["outward_axis_world"], dtype=np.float64
    )
    research = high["simulation_research_range"]
    position_delta = observed_position - reference_position
    axis_delta = float(
        math.acos(
            np.clip(
                float(observed_axis @ reference_axis)
                / float(
                    np.linalg.norm(observed_axis)
                    * np.linalg.norm(reference_axis)
                ),
                -1.0,
                1.0,
            )
        )
    )
    finite_range_pass = bool(
        abs(float(position_delta[0]))
        <= float(research["lateral_x_absolute_m"])
        and abs(float(position_delta[1]))
        <= float(research["lateral_y_absolute_m"])
        and abs(float(position_delta[2]))
        <= float(research["support_z_absolute_m"])
        and axis_delta <= float(research["axis_tilt_cone_rad"])
        and research["yaw_consumed_by_transport_target"] is False
    )
    range_audit = {
        "reference_provider_sha256": file_sha256(reference_path),
        "fresh_provider_sha256": file_sha256(provider_result_path),
        "fresh_minus_reference_position_xyz_m": position_delta.tolist(),
        "fresh_minus_reference_axis_angle_rad": axis_delta,
        "finite_research_set": research,
        "yaw_consumed": False,
        "online_object_truth_used": False,
        "pass": finite_range_pass,
    }
    if not finite_range_pass:
        _write_json_sealed(
            output / "high_motion_rejected.json",
            {
                "schema_version": "kcg_te_visual_high_reobserve_rejection_v1",
                "capture_id": arguments.capture_id,
                "reason": "FRESH_PROVIDER_OUTSIDE_FROZEN_FINITE_RESEARCH_SET",
                "range_audit": range_audit,
                "robot_command_count": 0,
                "truth_read": False,
            },
        )
        return 2

    target = build_visual_transport_target(
        provider_result_path=provider_result_path,
        relation_path=transport_relation_path,
        repository_root=repository,
    )
    target_path = output / "visual_transport_target.json"
    _write_json_sealed(target_path, target)
    # Reuse the existing target/grasp binding validator.  This assignment is
    # internal and happens only after the fresh provider and target are sealed.
    arguments.visual_transport_target = str(target_path)
    validated_target = _load_visual_transport_motion_input(
        repository, arguments, inputs, visual_grasp
    )
    if validated_target != target:
        raise RuntimeError("fresh visual target changed during validation")
    world_from_object = np.asarray(
        target["world_from_transport_object_row_major"], dtype=np.float64
    ).reshape(4, 4)
    motion_plan = control.build_joint_motion_plan(
        repository,
        inputs,
        visual_control_plan,
        world_from_object,
        include_lift=grasp_servo,
    )
    planned_target = np.asarray(
        motion_plan["world_from_hand_base_target"], dtype=np.float64
    ).reshape(4, 4)
    target_from_file = np.asarray(
        target["world_from_hand_base_target_row_major"], dtype=np.float64
    ).reshape(4, 4)
    if not np.allclose(
        planned_target, target_from_file, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("motion plan did not consume the fresh visual target")
    approach_direction = np.asarray(
        motion_plan["approach_direction_world"], dtype=np.float64
    )
    high_world_from_hand = planned_target.copy()
    high_clearance = float(
        high["static_path_panel"]["approach_high_clearance_m"]
    )
    high_world_from_hand[:3, 3] -= approach_direction * high_clearance
    high_arm = np.asarray(
        motion_plan["approach_arm_waypoints_rad"][0], dtype=np.float64
    )
    high_hand = np.asarray(
        motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
    )
    high_fk = np.asarray(
        inputs.robot_model.forward_kinematics(
            tuple(np.concatenate((high_arm, high_hand))),
            enforce_limits=False,
        )["handbase_link"],
        dtype=np.float64,
    )
    high_fk_position_error = float(
        np.linalg.norm(high_fk[:3, 3] - high_world_from_hand[:3, 3])
    )
    high_fk_rotation_error = _rotation_error_rad(
        high_fk[:3, :3], high_world_from_hand[:3, :3]
    )
    if (
        high_fk_position_error > 1.0e-5
        or high_fk_rotation_error > 1.0e-5
        or int(high["static_path_panel"]["state_count"]) != 126
        or float(
            high["static_path_panel"][
                "minimum_range_robust_plug_clearance_m"
            ]
        )
        <= 0.0
    ):
        raise RuntimeError("fresh approach-high target violates its static contract")
    plan_record = {
        "schema_version": "kcg_te_visual_high_motion_plan_v1",
        "capture_id": arguments.capture_id,
        "provider_result_sha256": file_sha256(provider_result_path),
        "visual_transport_target_sha256": file_sha256(target_path),
        "range_audit": range_audit,
        "static_path_panel": high["static_path_panel"],
        "world_from_hand_base_pregrasp_target_row_major": (
            planned_target.ravel().tolist()
        ),
        "world_from_hand_base_approach_high_row_major": (
            high_world_from_hand.ravel().tolist()
        ),
        "approach_high_joint_target_rad": high_arm.tolist(),
        "pregrasp_hand_joint_target_rad": high_hand.tolist(),
        "approach_high_fk_position_error_m": high_fk_position_error,
        "approach_high_fk_rotation_error_rad": high_fk_rotation_error,
        "motion_plan": _json_ready(motion_plan),
        "descent_waypoints_computed_by_existing_planner_but_not_authorized": (
            not grasp_servo
        ),
        "same_reset_visual_grasp_authorized_by_current_user_goal": grasp_servo,
        "descent_contact_closure_or_lift_commanded_at_plan_seal": False,
        "online_object_semantic_instance_or_contact_truth_used": False,
    }
    motion_plan_path = output / "fresh_visual_motion_plan.json"
    _write_json_sealed(motion_plan_path, plan_record)

    metadata = ft_articulation._articulation_view._metadata
    joint_indices = dict(metadata.joint_indices)
    ft_contract = high["wrist_ft_safety"]
    if ft_contract["source_joint"] not in joint_indices:
        raise RuntimeError("hand2arm is absent from physical reaction metadata")
    reaction_row = int(joint_indices[ft_contract["source_joint"]]) + 1
    reactions = _host_array(ft_articulation.get_measured_joint_forces())
    if (
        reactions.ndim != 2
        or reactions.shape[1] != 6
        or reaction_row >= len(reactions)
        or not np.all(np.isfinite(reactions))
    ):
        raise RuntimeError("physical hand2arm reaction row is unavailable")
    inertial_path = _bound_repository_file(
        repository,
        high["source_evidence"]["hand_inertial_source"],
        "frozen hand inertial source",
    )
    hand_inertials = _load_frozen_hand_inertials(inertial_path)
    if arguments.dynamic_inertia_compensation_enabled and not np.allclose(
        np.asarray(motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64),
        np.asarray(
            arguments.dynamic_inertia_contract[
                "frozen_pregrasp_hand_positions_rad"
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "dynamic-inertia frozen hand state differs from the motion plan"
        )

    command_api_counter: dict[str, int] = {}
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH,
        EXPECTED_DOF_NAMES,
        arguments.dynamic_settings,
        command_api_counter=command_api_counter,
    )
    initialization_api_counts = dict(command_api_counter)
    robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
    auditor = _HighObservationWristFtAuditor(
        ft_articulation=ft_articulation,
        reaction_row=reaction_row,
        robot_model=inputs.robot_model,
        hand_inertials=hand_inertials,
        gravity_m_s2=float(scene["gravity_m_s2"]),
        physics_dt_s=float(arguments.dynamic_settings["physics_dt_s"]),
        task_rotation_world=high_world_from_hand[:3, :3],
        force_limit_n=float(ft_contract["maximum_resultant_force_n"]),
        torque_limit_nm=float(ft_contract["maximum_resultant_torque_nm"]),
        dynamic_inertia_compensation_enabled=(
            arguments.dynamic_inertia_compensation_enabled
        ),
        frozen_hand_positions=motion_plan["pregrasp_hand_positions_rad"],
        dynamic_inertia_enabled_phases=(
            (
                "approach_above",
                "visual_servo_correction",
                "visual_servo_correction_settle",
                "visual_servo_descent",
                "visual_servo_pregrasp_hold",
                "visual_servo_recovery",
            )
            if grasp_servo
            else ("approach_above",)
            if arguments.visual_second_short_prefix_document is None
            else tuple(
                arguments.visual_second_short_prefix_document[
                    "wrist_ft_safety"
                ]["dynamic_inertia_enabled_phases"]
            )
        ),
    )
    if grasp_servo:
        truth_auditor = visual_grasp_services["truth_auditor"]
        ft_capture = auditor.capture

        def capture_ft_and_posthoc_truth(**kwargs):
            ft_capture(**kwargs)
            truth_auditor.capture(**kwargs)

        auditor.capture = capture_ft_and_posthoc_truth
    stepper = control.JointSignalStepper(
        robot=robot,
        world=world,
        auditor=auditor,
        active_indices=active_indices,
        arm_indices=arm_indices,
        arm_lower_limits=lower,
        arm_upper_limits=upper,
        settings=arguments.dynamic_settings,
        render=bool(arguments.gui),
        robot_model=inputs.robot_model if grasp_servo else None,
        payload_model=(
            visual_payload_model if grasp_servo else None
        ),
        command_api_counter=command_api_counter,
    )
    auditor.stepper = stepper
    dt = float(arguments.dynamic_settings["physics_dt_s"])
    tare_steps = round(float(ft_contract["tare_duration_s"]) / dt)
    home_arm = np.zeros(7, dtype=np.float64)
    home_hand = np.zeros(4, dtype=np.float64)
    for _ in stepper.active_steps(tare_steps):
        stepper.advance("ft_free_space_tare", home_arm, home_hand)
    if stepper.abort_reason is None:
        auditor.finalize_tare(int(ft_contract["minimum_tare_samples"]))
    pregrasp_result = None
    if stepper.abort_reason is None:
        pregrasp_result = control.run_pregrasp_sequence(
            stepper,
            motion_plan,
            arguments.dynamic_settings,
            stop_after_approach_high=True,
            observation_wait_hold_duration_s=float(
                high["motion"]["observation_wait_hold_duration_s"]
            ),
            approach_high_motion_duration_s=(
                None
                if arguments.visual_high_reobserve_variant_document is None
                else float(
                    arguments.approach_high_motion_duration_effective_s
                )
            ),
        )

    allowed_phases = set(high["motion"]["allowed_controller_phases"])
    observed_phases = set(auditor.phase_counts)
    forbidden_phases = sorted(observed_phases - allowed_phases)
    online_api_counts = {
        name: int(count) - int(initialization_api_counts.get(name, 0))
        for name, count in command_api_counter.items()
    }
    online_position_target_calls = int(
        online_api_counts.get("set_dof_position_targets", 0)
    )
    high_reached = bool(
        stepper.abort_reason is None
        and pregrasp_result is not None
        and pregrasp_result.get("high_wait_reached_without_abort") is True
        and int(pregrasp_result.get("descent_command_count", -1)) == 0
        and not forbidden_phases
        and online_position_target_calls == stepper.step_index
        and len(auditor.samples) == stepper.step_index
    )

    def recover_precontact_to_home(reason: str) -> dict[str, Any]:
        original_abort = stepper.abort_reason
        ft_triggered = auditor.first_ft_stop is not None
        if original_abort is not None and not ft_triggered:
            return {
                "requested": True,
                "started": False,
                "completed": False,
                "reason": reason,
                "original_abort_reason": original_abort,
                "failure_reason": "NON_FT_ABORT_NOT_CLEARED_FOR_RECOVERY",
            }
        reverse_targets = [
            np.asarray(row["active_targets_rad"], dtype=np.float64)
            for row in auditor.samples
        ]
        if not reverse_targets:
            return {
                "requested": True,
                "started": False,
                "completed": False,
                "reason": reason,
                "original_abort_reason": original_abort,
                "failure_reason": "NO_EXECUTED_TARGET_HISTORY",
            }
        if ft_triggered:
            auditor.begin_precontact_recovery()
        stepper.abort_reason = None
        open_hand = np.asarray(
            motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
        )
        executed = 0
        for target in reversed(reverse_targets):
            if stepper.abort_reason is not None:
                break
            stepper.advance(
                "visual_servo_recovery",
                np.asarray(target[:7], dtype=np.float64),
                open_hand,
            )
            executed += 1
        final_positions = (
            None if stepper.latest is None else np.asarray(stepper.latest[0])
        )
        final_home_error = (
            None
            if final_positions is None
            else float(np.max(np.abs(final_positions[:7])))
        )
        return {
            "requested": True,
            "started": True,
            "completed": bool(
                stepper.abort_reason is None
                and final_home_error is not None
                and final_home_error <= 0.02
            ),
            "reason": reason,
            "original_abort_reason": original_abort,
            "ft_triggered": ft_triggered,
            "reversed_recorded_target_count": len(reverse_targets),
            "executed_recovery_command_count": executed,
            "final_home_arm_error_rad": final_home_error,
            "failure_reason": stepper.abort_reason,
            "finger_target": "FROZEN_PREGRASP_OPENING",
            "object_or_contact_truth_used": False,
        }
    last_sample = None if not auditor.samples else auditor.samples[-1]
    arrival_position_error = None
    arrival_rotation_error = None
    arrival_joint_error = None
    if last_sample is not None:
        actual_hand = np.eye(4, dtype=np.float64)
        actual_hand[:3, :3] = np.asarray(
            last_sample["handbase_rotation_world_row_major"], dtype=np.float64
        )
        actual_hand[:3, 3] = np.asarray(
            last_sample["handbase_position_world_m"], dtype=np.float64
        )
        arrival_position_error = float(
            np.linalg.norm(
                actual_hand[:3, 3] - high_world_from_hand[:3, 3]
            )
        )
        arrival_rotation_error = _rotation_error_rad(
            actual_hand[:3, :3], high_world_from_hand[:3, :3]
        )
        arrival_joint_error = float(
            np.max(
                np.abs(
                    np.asarray(last_sample["active_positions_rad"][:7])
                    - high_arm
                )
            )
        )
    high_motion_report = {
        "schema_version": "kcg_te_visual_high_motion_execution_v1",
        "capture_id": arguments.capture_id,
        "reset_uuid": reset_uuid,
        "fresh_provider_result_sha256": file_sha256(provider_result_path),
        "fresh_visual_target_sha256": file_sha256(target_path),
        "fresh_motion_plan_sha256": file_sha256(motion_plan_path),
        "initialization_robot_api_counts": initialization_api_counts,
        "online_robot_api_counts": online_api_counts,
        "online_position_target_calls_equal_completed_steps": (
            online_position_target_calls == stepper.step_index
        ),
        "completed_physics_steps": stepper.step_index,
        "sample_count": len(auditor.samples),
        "phase_counts": dict(auditor.phase_counts),
        "forbidden_phase_counts": {
            phase: int(auditor.phase_counts.get(phase, 0))
            for phase in (
                "approach_descent",
                "pregrasp_hold",
                "finger_effort_tare",
                "finger_closure",
                "lift",
                "hold",
            )
        },
        "unexpected_phases": forbidden_phases,
        "native_drive_audit": _json_ready(drive_audit),
        "pregrasp_result": _json_ready(pregrasp_result),
        "approach_high_motion_duration": {
            "configured_s": float(
                arguments.approach_high_motion_duration_configured_s
            ),
            "effective_s": float(
                arguments.approach_high_motion_duration_effective_s
            ),
            "preshape_duration_s": float(
                arguments.dynamic_settings["approach_above_duration_s"]
            ),
            "single_variable_variant_applied": (
                arguments.visual_high_reobserve_variant_document is not None
            ),
        },
        "abort_reason": stepper.abort_reason,
        "maximum_joint_speed_rad_s": stepper.maximum_speed,
        "maximum_arm_tracking_error_rad": stepper.maximum_arm_error,
        "arrival_position_error_m": arrival_position_error,
        "arrival_rotation_error_rad": arrival_rotation_error,
        "arrival_maximum_arm_joint_error_rad": arrival_joint_error,
        "wrist_ft": auditor.summary(),
        "ft_limit_scope": (
            "USER_FROZEN_SIMULATION_ONLY_RESEARCH_SOFT_LIMIT_NOT_COMPLETE_"
            "WORKFLOW_A_CALIBRATION_OR_HARDWARE_LOAD_CERTIFICATION"
        ),
        "current_position_drive_target_held_after_stop": True,
        "descent_command_count": 0,
        "finger_closure_command_count": 0,
        "online_object_semantic_instance_or_contact_truth_used": False,
        "high_wait_reached": high_reached,
    }
    high_motion_path = output / "high_motion_execution.json"
    _write_json_sealed(high_motion_path, high_motion_report)
    samples_path = output / "high_motion_joint_ft_samples.json"
    _write_json_sealed(
        samples_path,
        {
            "schema_version": "kcg_te_visual_high_joint_ft_samples_v1",
            "capture_id": arguments.capture_id,
            "reset_uuid": reset_uuid,
            "physical_ft_source": "hand2arm",
            "online_object_semantic_instance_or_contact_truth_used": False,
            "samples": auditor.samples,
        },
    )
    if not high_reached:
        recovery = (
            recover_precontact_to_home("APPROACH_HIGH_FAILED")
            if grasp_servo
            else None
        )
        trace.update(
            {
                "motion_plan_constructed": True,
                "world_pregrasp_target_consumed": True,
                "robot_target_position_velocity_api_call_count": sum(
                    command_api_counter.values()
                ),
                "robot_motion_command_count": online_position_target_calls,
                "high_motion_execution_sha256": file_sha256(high_motion_path),
                "descent_command_count": 0,
                "finger_closure_command_count": 0,
                "high_wait_reached": False,
                "second_provider_ready": False,
                "formal_dynamic_pass": False,
                "control_authorized": False,
                "precontact_recovery": recovery,
            }
        )
        _write_json_sealed(output / "trace.json", trace)
        print(json.dumps(high_motion_report, ensure_ascii=False, indent=2))
        return 2

    def seal_postmotion_failure(reason: str, evidence: Mapping[str, object]) -> int:
        phase_counts_before_recovery = dict(auditor.phase_counts)
        last_phase_before_recovery = (
            None if not auditor.samples else str(auditor.samples[-1]["phase"])
        )
        recovery = (
            recover_precontact_to_home(reason) if grasp_servo else None
        )
        online_calls_now = int(
            command_api_counter.get("set_dof_position_targets", 0)
            - initialization_api_counts.get("set_dof_position_targets", 0)
        )
        descent_commands = sum(
            int(phase_counts_before_recovery.get(phase, 0))
            for phase in (
                "visual_servo_correction",
                "visual_servo_correction_settle",
                "visual_servo_descent",
                "pregrasp_hold",
            )
        )
        closure_commands = sum(
            count
            for phase, count in phase_counts_before_recovery.items()
            if phase.startswith("finger_") or phase in {
                "tare",
                "preload",
                "prelift_effort_check",
            }
        )
        ft_failure_samples_path = output / "precontact_failure_joint_ft_samples.json"
        _write_json_sealed(
            ft_failure_samples_path,
            {
                "schema_version": "kcg_te_visual_servo_precontact_ft_samples_v1",
                "failure_reason": reason,
                "last_phase_before_recovery": last_phase_before_recovery,
                "physical_source": "hand2arm",
                "summary": auditor.summary(),
                "samples": auditor.samples,
                "online_object_semantic_instance_or_contact_truth_used": False,
            },
        )
        truth_failure_samples_path = (
            output / "precontact_failure_isolated_truth_samples.json"
        )
        if grasp_servo:
            _write_json_sealed(
                truth_failure_samples_path,
                {
                    "schema_version": (
                        "kcg_te_visual_servo_precontact_isolated_truth_samples_v1"
                    ),
                    "failure_reason": reason,
                    "evaluation_time": "AFTER_CONTROL_AND_RECOVERY_FINISHED",
                    "returned_to_controller": False,
                    "samples": visual_grasp_services["truth_auditor"].samples,
                },
            )
        trace.update(
            {
                "motion_plan_constructed": True,
                "world_pregrasp_target_consumed": True,
                "robot_target_position_velocity_api_call_count": sum(
                    command_api_counter.values()
                ),
                "robot_motion_command_count": online_calls_now,
                "high_motion_execution_sha256": file_sha256(high_motion_path),
                "high_motion_samples_sha256": file_sha256(samples_path),
                "descent_command_count": descent_commands,
                "finger_closure_command_count": closure_commands,
                "online_object_or_contact_truth_used": False,
                "truth_audit_data_returned_to_controller": False,
                "high_wait_reached": True,
                "second_provider_ready": False,
                "formal_dynamic_pass": False,
                "control_authorized": False,
                "precontact_recovery": recovery,
                "truth_audit_read_during_run_for_isolated_recording": (
                    grasp_servo
                ),
                "precontact_failure_ft_samples_sha256": file_sha256(
                    ft_failure_samples_path
                ),
                "precontact_failure_truth_samples_sha256": (
                    file_sha256(truth_failure_samples_path)
                    if grasp_servo
                    else None
                ),
            }
        )
        failure = {
            "schema_version": "kcg_te_visual_high_reobserve_result_v1",
            "status": "IMPLEMENTING",
            "physical_result": "VISUAL_SERVO_PRECONTACT_FAILURE_AND_RECOVERY",
            "failure_reason": reason,
            "failure_evidence": dict(evidence),
            "high_wait_reached": True,
            "second_provider_ready": False,
            "last_phase_before_recovery": last_phase_before_recovery,
            "robot_motion_command_count": online_calls_now,
            "descent_command_count": descent_commands,
            "finger_closure_command_count": closure_commands,
            "contact_grasp_insertion_or_locking_occurred": False,
            "truth_read": False,
            "hardware_authorized": False,
            "precontact_recovery": recovery,
            "precontact_failure_ft_samples": str(
                ft_failure_samples_path.relative_to(repository)
            ),
            "precontact_failure_isolated_truth_samples": (
                str(truth_failure_samples_path.relative_to(repository))
                if grasp_servo
                else None
            ),
        }
        _write_json_sealed(output / "trace.json", trace)
        _write_json_sealed(output / "result.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2

    # The drive remains on the reached high target.  No new robot command or
    # physics step is issued from this point through the second provider seal.
    counter_before_reobservation = dict(command_api_counter)
    positions_before = _host_array(robot.get_dof_positions(indices=0))[0]
    velocities_before = _host_array(robot.get_dof_velocities(indices=0))[0]
    reobserve_root = output / "reobservation"
    reobserve_observation = reobserve_root / "observation"
    reobserve_started = datetime.now(timezone.utc)
    capture = capture_d38999_rgbd_raw_formal(
        bindings=bindings,
        simulation_app=simulation_app,
        world=world,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
        output_dir=reobserve_observation,
        camera_clipping_range_m=tuple(camera["clipping_range_m"]),
        rt_subframes=int(
            arguments.same_reset_rgbd_document["timing"][
                "capture_rt_subframes"
            ]
        ),
    )
    positions_after = _host_array(robot.get_dof_positions(indices=0))[0]
    velocities_after = _host_array(robot.get_dof_velocities(indices=0))[0]
    if capture.passed is not True or capture.rgb is None or capture.depth is None:
        _write_json_sealed(
            reobserve_root / "capture_failure.json",
            {
                "schema_version": "kcg_te_visual_high_reobserve_failure_v1",
                "capture_id": arguments.capture_id,
                "reset_uuid": reset_uuid,
                "reason": "SECOND_RAW_RGBD_CAPTURE_FAILED",
                "robot_api_counts_unchanged": (
                    command_api_counter == counter_before_reobservation
                ),
                "truth_read": False,
                "raw_capture_metrics": capture.metrics,
            },
        )
        return seal_postmotion_failure(
            "SECOND_RAW_RGBD_CAPTURE_FAILED",
            {
                "local_failure_path": str(
                    (reobserve_root / "capture_failure.json").relative_to(
                        repository
                    )
                ),
                "robot_api_counts_unchanged": (
                    command_api_counter == counter_before_reobservation
                ),
            },
        )
    if command_api_counter != counter_before_reobservation:
        return seal_postmotion_failure(
            "ROBOT_COMMAND_API_CHANGED_DURING_REOBSERVATION",
            {
                "before": counter_before_reobservation,
                "after": dict(command_api_counter),
                "truth_read": False,
            },
        )

    second_observability = evaluate_te_rgbd_observability(
        rgb=capture.rgb,
        depth_m=capture.depth,
        static_depth_m=static_depth,
        intrinsics=observed_intrinsics,
        world_from_camera=expected_world_from_camera,
        workspaces=(plug_workspace,),
        minimum_key_width_px=4.0,
        minimum_foreground_depth_delta_m=float(
            arguments.same_reset_rgbd_document["image_gates"][
                "minimum_foreground_depth_delta_m"
            ]
        ),
    )
    second_input = copy.deepcopy(provider_input)
    second_capture_id = arguments.capture_id + "__high_reobserve"
    second_input["capture_id"] = second_capture_id
    second_input["capture_contract_sha256"] = file_sha256(high_config_path)
    second_input["capture_time"] = {
        "clock_domain": "host_utc",
        "capture_started_at_utc": reobserve_started.isoformat(),
        "observed_frame_timestamp_utc": capture.metrics[
            "capture_timestamp_utc"
        ],
        "static_frame_timestamp_utc": provider_input["capture_time"][
            "static_frame_timestamp_utc"
        ],
    }
    second_input["same_reset_evidence"] = {
        "reset_uuid": reset_uuid,
        "reset_count": 1,
        "resets_after_fresh_frame": 0,
        "robot_target_position_velocity_api_call_count": sum(
            command_api_counter.values()
        ),
        "robot_motion_command_count": online_position_target_calls,
        "settle_physics_steps_before_frame": stepper.step_index,
        "provider_truth_read_before_result": False,
    }
    second_input["ordinary_rgb"] = {
        "encoding": "png_rgb_uint8",
        "path_relative_to_manifest": "observation/rgb.png",
        "sha256": file_sha256(reobserve_observation / "rgb.png"),
        "shape": list(capture.rgb.shape),
    }
    second_input["ordinary_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "observation/depth_m.npy",
        "sha256": file_sha256(reobserve_observation / "depth_m.npy"),
        "shape": list(capture.depth.shape),
    }
    second_background_dir = reobserve_root / "background"
    second_background_dir.mkdir(parents=False, exist_ok=False)
    second_background_path = second_background_dir / "depth_m.npy"
    shutil.copyfile(background_path, second_background_path)
    if file_sha256(second_background_path) != file_sha256(background_path):
        raise RuntimeError("second-observation frozen background copy differs")
    second_input["ordinary_static_scene_depth"] = {
        "encoding": "npy_distance_to_image_plane_m_float32",
        "path_relative_to_manifest": "background/depth_m.npy",
        "sha256": file_sha256(second_background_path),
        "shape": list(static_depth.shape),
        "frozen_source_path": provider_input["ordinary_static_scene_depth"][
            "frozen_source_path"
        ],
        "frozen_source_sha256": provider_input[
            "ordinary_static_scene_depth"
        ]["frozen_source_sha256"],
    }
    second_input["observability"] = second_observability
    second_input["control_authorized"] = False
    second_input["pose_result"] = None
    second_input_path = reobserve_root / "provider_input.json"
    _write_json_sealed(second_input_path, second_input)
    try:
        second_result = run_te_rgbd_pose_provider(
            second_input_path, repository
        )
    except Exception as error:
        _write_json_sealed(
            reobserve_root / "provider_failure.json",
            {
                "schema_version": "kcg_te_visual_high_reobserve_failure_v1",
                "capture_id": second_capture_id,
                "reset_uuid": reset_uuid,
                "reason": "SECOND_PROVIDER_FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "provider_input_sha256": file_sha256(second_input_path),
                "robot_api_counts_unchanged": (
                    command_api_counter == counter_before_reobservation
                ),
                "truth_read": False,
            },
        )
        return seal_postmotion_failure(
            "SECOND_PROVIDER_FAILED",
            {
                "local_failure_path": str(
                    (reobserve_root / "provider_failure.json").relative_to(
                        repository
                    )
                ),
                "provider_input_sha256": file_sha256(second_input_path),
                "robot_api_counts_unchanged": (
                    command_api_counter == counter_before_reobservation
                ),
            },
        )
    second_result["provider_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    second_result_path = reobserve_root / "pose_provider_result.json"
    _write_json_sealed(second_result_path, second_result)
    second_sealed_at = datetime.now(timezone.utc)
    second_frame_time = datetime.fromisoformat(
        capture.metrics["capture_timestamp_utc"]
    )
    second_frame_to_seal = (second_sealed_at - second_frame_time).total_seconds()
    second_ring = second_result["transport_grasp_pose"]["derivation"].get(
        "rx180_axisymmetric_ring"
    )
    image_gates = arguments.same_reset_rgbd_document["image_gates"]
    second_image_ready = bool(
        second_result.get("provider_scope") == "TRANSPORT_PLUG_ONLY"
        and second_result["transport_grasp_pose"].get("status")
        == "OBSERVED_AXIS_POSITION_YAW_FREE"
        and isinstance(second_ring, Mapping)
        and second_ring.get("fit_valid") is True
        and int(second_ring.get("point_count", 0))
        >= int(image_gates["minimum_rx180_ring_points"])
        and float(image_gates["minimum_ring_radius_m"])
        <= float(second_ring.get("radius_m", -1.0))
        <= float(image_gates["maximum_ring_radius_m"])
        and float(second_ring.get("absolute_radial_residual_p99_m", np.inf))
        <= float(image_gates["maximum_ring_absolute_residual_p99_m"])
        and second_frame_to_seal
        <= float(
            arguments.same_reset_rgbd_document["timing"][
                "maximum_frame_to_provider_seal_s"
            ]
        )
        and second_result["transport_grasp_pose"].get(
            "transport_grasp_control_allowed"
        )
        is False
    )
    second_transport = second_result["transport_grasp_pose"]
    first_second_consistency = {
        "evaluated": False,
        "position_delta_xyz_m": None,
        "axis_angle_rad": None,
        "maximum_absolute_position_delta_xyz_m": [
            2.0 * float(research["lateral_x_absolute_m"]),
            2.0 * float(research["lateral_y_absolute_m"]),
            2.0 * float(research["support_z_absolute_m"]),
        ],
        "maximum_axis_angle_rad": 2.0 * float(research["axis_tilt_cone_rad"]),
        "pass": False,
        "object_truth_used": False,
    }
    if second_image_ready:
        second_candidate_position = np.asarray(
            second_transport["position_xyz_m"], dtype=np.float64
        )
        second_candidate_axis = np.asarray(
            second_transport["outward_axis_world"], dtype=np.float64
        )
        consistency_delta = second_candidate_position - observed_position
        consistency_axis_angle = float(
            math.acos(
                np.clip(
                    float(second_candidate_axis @ observed_axis)
                    / float(
                        np.linalg.norm(second_candidate_axis)
                        * np.linalg.norm(observed_axis)
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        consistency_pass = bool(
            abs(float(consistency_delta[0]))
            <= 2.0 * float(research["lateral_x_absolute_m"])
            and abs(float(consistency_delta[1]))
            <= 2.0 * float(research["lateral_y_absolute_m"])
            and abs(float(consistency_delta[2]))
            <= 2.0 * float(research["support_z_absolute_m"])
            and consistency_axis_angle
            <= 2.0 * float(research["axis_tilt_cone_rad"])
        )
        first_second_consistency.update(
            {
                "evaluated": True,
                "position_delta_xyz_m": consistency_delta.tolist(),
                "axis_angle_rad": consistency_axis_angle,
                "pass": consistency_pass,
            }
        )
    command_counts_unchanged = bool(
        command_api_counter == counter_before_reobservation
    )
    joint_position_delta = positions_after - positions_before
    paused_joint_readback_atol_rad = float(
        4.0 * np.finfo(np.float32).eps
    )
    joint_positions_unchanged = bool(
        np.all(np.isfinite(joint_position_delta))
        and np.max(np.abs(joint_position_delta))
        <= paused_joint_readback_atol_rad
    )
    timeline_paused_for_capture = bool(
        capture.metrics.get("timeline_pause", {}).get("paused_for_capture")
        is True
    )
    second_ready = bool(
        second_image_ready
        and command_counts_unchanged
        and joint_positions_unchanged
        and timeline_paused_for_capture
        and first_second_consistency["pass"] is True
    )
    reobserve_report = {
        "schema_version": "kcg_te_visual_high_reobserve_sealed_v1",
        "capture_id": second_capture_id,
        "reset_uuid": reset_uuid,
        "world_reset_count": 1,
        "same_reset_as_first_provider": True,
        "robot_command_api_counts_before_capture": counter_before_reobservation,
        "robot_command_api_counts_after_provider_seal": dict(
            command_api_counter
        ),
        "robot_api_counts_unchanged_during_capture_and_provider": (
            command_counts_unchanged
        ),
        "joint_position_change_during_paused_capture_rad": (
            joint_position_delta
        ).tolist(),
        "joint_position_stationarity_gate": {
            "method": "FLOAT32_READBACK_FOUR_EPSILON",
            "absolute_tolerance_rad": paused_joint_readback_atol_rad,
            "maximum_absolute_delta_rad": float(
                np.max(np.abs(joint_position_delta))
            ),
            "pass": joint_positions_unchanged,
        },
        "joint_velocity_before_capture_rad_s": velocities_before.tolist(),
        "joint_velocity_after_capture_rad_s": velocities_after.tolist(),
        "joint_positions_unchanged_during_paused_capture": (
            joint_positions_unchanged
        ),
        "timeline_paused_for_capture": timeline_paused_for_capture,
        "raw_capture_metrics": capture.metrics,
        "observability": second_observability,
        "provider_input_sha256": file_sha256(second_input_path),
        "provider_result_sha256": file_sha256(second_result_path),
        "provider_result_sealed_before_truth_read": True,
        "provider_sealed_at_utc": second_sealed_at.isoformat(),
        "frame_to_provider_seal_s": second_frame_to_seal,
        "provider_image_gate_pass": second_image_ready,
        "first_second_truth_free_consistency": first_second_consistency,
        "result": (
            "READY_FOR_NEXT_SHORT_PATH_PLANNING"
            if second_ready
            else "REOBSERVATION_MISS_STOP_NO_DESCENT"
        ),
        "second_provider_ready": second_ready,
        "descent_command_count": 0,
        "truth_read": False,
        "control_authorized": False,
    }
    reobserve_report_path = reobserve_root / "sealed_reobservation_report.json"
    _write_json_sealed(reobserve_report_path, reobserve_report)

    if grasp_servo and not second_ready:
        return seal_postmotion_failure(
            "SECOND_RGBD_NOT_READY_FOR_VISUAL_SERVO",
            {
                "sealed_reobservation_report_sha256": file_sha256(
                    reobserve_report_path
                ),
                "truth_read": False,
            },
        )

    if grasp_servo:
        try:
            second_target = build_visual_transport_target(
                provider_result_path=second_result_path,
                relation_path=transport_relation_path,
                repository_root=repository,
            )
        except Exception as error:
            return seal_postmotion_failure(
                "SECOND_VISUAL_TARGET_BUILD_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        second_target_path = reobserve_root / "servo_second_visual_target.json"
        _write_json_sealed(second_target_path, second_target)
        try:
            loaded_second_target, _ = load_visual_transport_target(
                second_target_path, repository
            )
        except Exception as error:
            return seal_postmotion_failure(
                "SECOND_VISUAL_TARGET_VALIDATION_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        if loaded_second_target != second_target:
            return seal_postmotion_failure(
                "SECOND_VISUAL_TARGET_CHANGED_AFTER_SEAL",
                {"truth_read": False},
            )
        second_world_from_object = np.asarray(
            second_target["world_from_transport_object_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        try:
            second_motion_plan = control.build_joint_motion_plan(
                repository,
                inputs,
                visual_control_plan,
                second_world_from_object,
                include_lift=True,
            )
        except Exception as error:
            return seal_postmotion_failure(
                "SECOND_VISUAL_MOTION_PLAN_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        second_pregrasp_target = np.asarray(
            second_motion_plan["world_from_hand_base_target"],
            dtype=np.float64,
        ).reshape(4, 4)
        if not np.allclose(
            second_pregrasp_target,
            np.asarray(
                second_target["world_from_hand_base_target_row_major"],
                dtype=np.float64,
            ).reshape(4, 4),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return seal_postmotion_failure(
                "SECOND_RGBD_TARGET_NOT_CONSUMED_BY_SERVO_PLAN",
                {"truth_read": False},
            )

        second_capture_active = np.asarray(
            positions_after[active_indices], dtype=np.float64
        )
        frozen_hand = np.asarray(
            second_motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
        )
        if (
            second_capture_active.shape != (11,)
            or not np.all(np.isfinite(second_capture_active))
            or not np.allclose(
                second_capture_active[7:], frozen_hand, rtol=0.0, atol=0.02
            )
        ):
            return seal_postmotion_failure(
                "SECOND_RGBD_ROBOT_STATE_NOT_FROZEN_OPEN_HAND",
                {"truth_read": False},
            )
        second_actual_hand = np.asarray(
            inputs.robot_model.forward_kinematics(
                tuple(second_capture_active), enforce_limits=False
            )["handbase_link"],
            dtype=np.float64,
        )
        error_before_vector = (
            second_pregrasp_target[:3, 3] - second_actual_hand[:3, 3]
        )
        error_before_norm = float(np.linalg.norm(error_before_vector))
        rotation_error_before = _rotation_error_rad(
            second_actual_hand[:3, :3], second_pregrasp_target[:3, :3]
        )
        correction_target = second_pregrasp_target.copy()
        correction_target[:3, 3] = (
            second_actual_hand[:3, 3] + 0.5 * error_before_vector
        )
        try:
            (
                correction_arm,
                correction_position_error,
                correction_rotation_error,
                _,
            ) = control.solve_bounded_hand_base_ik(
                    inputs.config.section("ik")["solver"],
                    model=inputs.robot_model,
                    hand_positions=frozen_hand,
                    target_world_from_hand_base=correction_target,
                    seed_arm_positions=(second_capture_active[:7],),
                    label="TE_VISUAL_SERVO_HALF_PREGRASP_ERROR",
                )
        except Exception as error:
            return seal_postmotion_failure(
                "HALF_ERROR_VISUAL_SERVO_IK_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        correction_arm = np.asarray(correction_arm, dtype=np.float64)
        if (
            correction_arm.shape != (7,)
            or not np.all(np.isfinite(correction_arm))
            or correction_position_error > 1.0e-5
            or correction_rotation_error > 1.0e-5
        ):
            return seal_postmotion_failure(
                "HALF_ERROR_VISUAL_SERVO_IK_OUTSIDE_TOLERANCE",
                {
                    "position_error_m": float(correction_position_error),
                    "rotation_error_rad": float(correction_rotation_error),
                    "truth_read": False,
                },
            )
        second_provider_age_s = (
            datetime.now(timezone.utc) - second_frame_time
        ).total_seconds()
        if second_provider_age_s > float(
            arguments.same_reset_rgbd_document["timing"][
                "maximum_frame_to_provider_seal_s"
            ]
        ):
            return seal_postmotion_failure(
                "SECOND_PROVIDER_STALE_BEFORE_CORRECTION_COMMAND",
                {
                    "provider_age_s": second_provider_age_s,
                    "truth_read": False,
                },
            )
        correction_steps = round(
            0.5
            * float(arguments.dynamic_settings["approach_descent_duration_s"])
            / dt
        )
        correction_start = second_capture_active[:7].copy()
        correction_calls_before = int(
            command_api_counter.get("set_dof_position_targets", 0)
        )
        for index in stepper.active_steps(correction_steps):
            blend = control.minimum_jerk_blend((index + 1) / correction_steps)
            stepper.advance(
                "visual_servo_correction",
                (1.0 - blend) * correction_start + blend * correction_arm,
                frozen_hand,
            )
        settled_count = 0
        correction_settled = False
        settle_steps = round(
            float(arguments.dynamic_settings["approach_above_settle_timeout_s"])
            / dt
        )
        for _ in stepper.active_steps(settle_steps):
            latest = stepper.advance(
                "visual_servo_correction_settle",
                correction_arm,
                frozen_hand,
            )
            if latest is None:
                break
            correction_joint_error = float(
                np.max(np.abs(np.asarray(latest[0][:7]) - correction_arm))
            )
            settled_count = (
                settled_count + 1
                if correction_joint_error
                <= float(
                    arguments.dynamic_settings[
                        "approach_above_settle_error_rad"
                    ]
                )
                else 0
            )
            if settled_count >= int(
                arguments.dynamic_settings[
                    "approach_above_settle_consecutive_samples"
                ]
            ):
                correction_settled = True
                break
        if stepper.abort_reason is not None or not correction_settled:
            return seal_postmotion_failure(
                "VISUAL_SERVO_CORRECTION_ABORTED_OR_NOT_SETTLED",
                {
                    "abort_reason": stepper.abort_reason,
                    "correction_settled": correction_settled,
                    "truth_read": False,
                },
            )

        try:
            third = _capture_third_visual_servo_rgbd(
                repository=repository,
                arguments=arguments,
                world=world,
                stage=stage,
                simulation_app=simulation_app,
                tabletop=tabletop,
                rgbd=rgbd,
                camera=camera,
                bindings=bindings,
                base_provider_input=provider_input,
                static_depth=static_depth,
                background_path=background_path,
                plug_workspace=plug_workspace,
                plug_workspace_record=plug_workspace_record,
                expected_world_from_camera=expected_world_from_camera,
                observed_intrinsics=observed_intrinsics,
                output=output,
                reset_uuid=reset_uuid,
                command_api_counter=command_api_counter,
                robot=robot,
                physics_step_index=stepper.step_index,
            )
        except Exception as error:
            return seal_postmotion_failure(
                "THIRD_RGBD_INFRASTRUCTURE_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        second_frame_timestamp = datetime.fromisoformat(
            capture.metrics["capture_timestamp_utc"]
        )
        third_frame_timestamp = (
            None
            if third.get("frame_timestamp_utc") is None
            else datetime.fromisoformat(str(third["frame_timestamp_utc"]))
        )
        timestamps_monotonic = bool(
            third_frame_timestamp is not None
            and third_frame_timestamp > second_frame_timestamp
        )
        if third.get("ready") is not True or not timestamps_monotonic:
            return seal_postmotion_failure(
                "THIRD_RGBD_MISS_AFTER_FINITE_CORRECTION",
                {
                    "third_status": third.get("status"),
                    "timestamps_monotonic": timestamps_monotonic,
                    "truth_read": False,
                },
            )

        third_result = third["provider_result"]
        third_result_path = Path(third["provider_result_path"])
        try:
            third_target = build_visual_transport_target(
                provider_result_path=third_result_path,
                relation_path=transport_relation_path,
                repository_root=repository,
            )
        except Exception as error:
            return seal_postmotion_failure(
                "THIRD_VISUAL_TARGET_BUILD_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        third_target_path = output / "third_observation" / "visual_target.json"
        _write_json_sealed(third_target_path, third_target)
        third_world_from_object = np.asarray(
            third_target["world_from_transport_object_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        try:
            third_motion_plan = control.build_joint_motion_plan(
                repository,
                inputs,
                visual_control_plan,
                third_world_from_object,
                include_lift=True,
            )
        except Exception as error:
            return seal_postmotion_failure(
                "THIRD_VISUAL_MOTION_PLAN_FAILED",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "truth_read": False,
                },
            )
        third_pregrasp_target = np.asarray(
            third_motion_plan["world_from_hand_base_target"],
            dtype=np.float64,
        ).reshape(4, 4)
        if not np.allclose(
            third_pregrasp_target,
            np.asarray(
                third_target["world_from_hand_base_target_row_major"],
                dtype=np.float64,
            ).reshape(4, 4),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return seal_postmotion_failure(
                "THIRD_RGBD_TARGET_NOT_CONSUMED_BY_FINAL_PLAN",
                {"truth_read": False},
            )
        third_full_positions = np.asarray(
            third["joint_positions_after_capture_rad"], dtype=np.float64
        )
        third_active = third_full_positions[active_indices]
        third_actual_hand = np.asarray(
            inputs.robot_model.forward_kinematics(
                tuple(third_active), enforce_limits=False
            )["handbase_link"],
            dtype=np.float64,
        )
        error_after_vector = (
            third_pregrasp_target[:3, 3] - third_actual_hand[:3, 3]
        )
        error_after_norm = float(np.linalg.norm(error_after_vector))
        rotation_error_after = _rotation_error_rad(
            third_actual_hand[:3, :3], third_pregrasp_target[:3, :3]
        )
        third_transport = third_result["transport_grasp_pose"]
        second_third_position_delta = (
            np.asarray(third_transport["position_xyz_m"], dtype=np.float64)
            - np.asarray(second_transport["position_xyz_m"], dtype=np.float64)
        )
        second_axis = np.asarray(
            second_transport["outward_axis_world"], dtype=np.float64
        )
        third_axis = np.asarray(
            third_transport["outward_axis_world"], dtype=np.float64
        )
        second_third_axis_angle = float(
            math.acos(
                np.clip(
                    float(second_axis @ third_axis)
                    / float(np.linalg.norm(second_axis) * np.linalg.norm(third_axis)),
                    -1.0,
                    1.0,
                )
            )
        )
        second_third_consistent = bool(
            abs(float(second_third_position_delta[0]))
            <= 2.0 * float(research["lateral_x_absolute_m"])
            and abs(float(second_third_position_delta[1]))
            <= 2.0 * float(research["lateral_y_absolute_m"])
            and abs(float(second_third_position_delta[2]))
            <= 2.0 * float(research["support_z_absolute_m"])
            and second_third_axis_angle
            <= 2.0 * float(research["axis_tilt_cone_rad"])
        )
        provider_age_s = (
            datetime.now(timezone.utc)
            - third_frame_timestamp
        ).total_seconds()
        error_decreased = bool(
            np.all(np.isfinite(error_before_vector))
            and np.all(np.isfinite(error_after_vector))
            and math.isfinite(error_before_norm)
            and math.isfinite(error_after_norm)
            and error_after_norm < error_before_norm
            and rotation_error_after <= rotation_error_before + 0.001
        )
        servo_record = {
            "schema_version": "kcg_te_visual_servo_error_decrease_v1",
            "reset_uuid": reset_uuid,
            "camera_calibration_id": camera["calibration_id"],
            "second_provider_sha256": file_sha256(second_result_path),
            "second_target_sha256": file_sha256(second_target_path),
            "third_provider_sha256": file_sha256(third_result_path),
            "third_target_sha256": file_sha256(third_target_path),
            "relation_path": str(transport_relation_path.relative_to(repository)),
            "relation_sha256": file_sha256(transport_relation_path),
            "error_before_correction_world_m": error_before_vector.tolist(),
            "error_before_correction_norm_m": error_before_norm,
            "correction_fraction": 0.5,
            "correction_world_target_position_m": (
                correction_target[:3, 3].tolist()
            ),
            "correction_command_count": int(
                command_api_counter.get("set_dof_position_targets", 0)
                - correction_calls_before
            ),
            "error_after_correction_world_m": error_after_vector.tolist(),
            "error_after_correction_norm_m": error_after_norm,
            "rotation_error_before_rad": rotation_error_before,
            "rotation_error_after_rad": rotation_error_after,
            "second_third_position_delta_m": (
                second_third_position_delta.tolist()
            ),
            "second_third_axis_angle_rad": second_third_axis_angle,
            "second_third_truth_free_consistent": second_third_consistent,
            "second_frame_timestamp_utc": second_frame_timestamp.isoformat(),
            "third_frame_timestamp_utc": third_frame_timestamp.isoformat(),
            "frame_timestamps_monotonic": timestamps_monotonic,
            "third_provider_age_before_remaining_motion_s": provider_age_s,
            "maximum_provider_age_s": float(
                arguments.same_reset_rgbd_document["timing"][
                    "maximum_frame_to_provider_seal_s"
                ]
            ),
            "position_error_decreased": error_decreased,
            "online_object_semantic_instance_or_contact_truth_used": False,
            "truth_audit_recorded_during_run_but_never_returned_to_control": True,
            "truth_evaluated_only_after_control_finished": True,
        }
        servo_record_path = output / "visual_servo_error_decrease.json"
        _write_json_sealed(servo_record_path, servo_record)
        if not (
            error_decreased
            and second_third_consistent
            and provider_age_s
            <= float(
                arguments.same_reset_rgbd_document["timing"][
                    "maximum_frame_to_provider_seal_s"
                ]
            )
        ):
            return seal_postmotion_failure(
                "VISUAL_SERVO_ERROR_DID_NOT_DECREASE_OR_THIRD_FRAME_STALE",
                {
                    "servo_record_sha256": file_sha256(servo_record_path),
                    "truth_read": False,
                },
            )

        approach = np.asarray(
            third_motion_plan["approach_arm_waypoints_rad"], dtype=np.float64
        )
        current_arm = third_active[:7].copy()
        nearest_index = int(
            np.argmin(np.linalg.norm(approach - current_arm, axis=1))
        )
        remaining = [current_arm]
        remaining.extend(approach[nearest_index + 1 :])
        if len(remaining) == 1 or not np.array_equal(remaining[-1], approach[-1]):
            remaining.append(approach[-1])
        remaining_waypoints = np.asarray(remaining, dtype=np.float64)
        descent_steps = round(
            float(arguments.dynamic_settings["approach_descent_duration_s"]) / dt
        )
        for index in stepper.active_steps(descent_steps):
            blend = control.minimum_jerk_blend((index + 1) / descent_steps)
            stepper.advance(
                "visual_servo_descent",
                control.piecewise_waypoint(remaining_waypoints, blend),
                frozen_hand,
            )
        pregrasp_settled_count = 0
        for _ in stepper.active_steps(
            round(float(arguments.dynamic_settings["pregrasp_hold_duration_s"]) / dt)
        ):
            latest = stepper.advance(
                "pregrasp_hold", approach[-1], frozen_hand
            )
            if latest is not None:
                hold_error = float(
                    np.max(np.abs(np.asarray(latest[0][:7]) - approach[-1]))
                )
                pregrasp_settled_count = (
                    pregrasp_settled_count + 1
                    if hold_error
                    <= float(
                        arguments.dynamic_settings[
                            "approach_above_settle_error_rad"
                        ]
                    )
                    else 0
                )
        pregrasp_tracking_error = float(
            np.max(np.abs(np.asarray(stepper.latest[0][:7]) - approach[-1]))
        )
        if (
            stepper.abort_reason is not None
            or pregrasp_tracking_error
            > float(
                arguments.dynamic_settings["approach_above_settle_error_rad"]
            )
            or pregrasp_settled_count
            < int(
                arguments.dynamic_settings[
                    "approach_above_settle_consecutive_samples"
                ]
            )
        ):
            return seal_postmotion_failure(
                "PRECONTACT_DESCENT_WRIST_FT_JOINT_OR_TRACKING_ABORT",
                {
                    "abort_reason": stepper.abort_reason,
                    "pregrasp_tracking_error_rad": pregrasp_tracking_error,
                    "pregrasp_settled_consecutive_samples": (
                        pregrasp_settled_count
                    ),
                    "servo_record_sha256": file_sha256(servo_record_path),
                    "truth_read": False,
                },
            )

        dynamic_for_grasp = dict(arguments.dynamic_settings)
        dynamic_for_grasp["contact_velocity_absolute_max_rad_s"] = float(
            transport_relation["hand_contract"][
                "contact_confirmation_velocity_absolute_max_rad_s"
            ]
        )
        pregrasp_for_grasp = {
            "arm": approach[-1].copy(),
            "hand": frozen_hand.copy(),
            "above_settled": True,
            "above_final_error_rad": pregrasp_tracking_error,
            "visual_tracking_stopped_before_first_expected_finger_contact": True,
        }
        grasp_result = control.run_grasp_lift_sequence(
            stepper,
            third_motion_plan,
            dynamic_for_grasp,
            pregrasp_for_grasp,
        )
        postcontact_failure_action = (
            None
            if grasp_result.get("failure_reason") is None
            and stepper.abort_reason is None
            else (
                "STOP_SIMULATION_WITH_CURRENT_BOUNDED_DRIVE_TARGETS_HELD_"
                "NO_AUTOMATIC_RELEASE_OR_DROP"
            )
        )
        outcome = control.controller_outcome(
            stepper,
            mode="grasp-lift",
            native_drive_audit=drive_audit,
            pregrasp=pregrasp_for_grasp,
            grasp=grasp_result,
        )
        arguments.visual_transport_target_consumption = {
            "provided": True,
            "motion_plan_pose_source": (
                "THIRD_ORDINARY_RGBD_AFTER_FINITE_ERROR_REDUCING_CORRECTION"
            ),
            "target_binding": {
                "path": str(third_target_path.relative_to(repository)),
                "sha256": file_sha256(third_target_path),
            },
            "world_from_transport_object_row_major": (
                third_target["world_from_transport_object_row_major"]
            ),
            "planned_world_from_hand_base_target_row_major": (
                third_pregrasp_target.ravel().tolist()
            ),
            "planned_approach_direction_world": list(
                third_motion_plan["approach_direction_world"]
            ),
            "target_matrix_matches_frozen_composition": True,
            "assembly_key_pose_consumed": False,
            "scene_contract_pose_used_for_motion_plan": False,
            "online_object_or_semantic_truth_used": False,
            "controller_execution_observed": True,
        }
        composer_path = (
            repository
            / "src/kcg_connector/kcg_connector/te_transport_grasp_target.py"
        ).resolve()
        arguments.visual_transport_target = str(third_target_path)
        arguments.visual_transport_target_path = third_target_path
        arguments.visual_transport_target_binding = {
            "target_path": str(third_target_path.relative_to(repository)),
            "target_sha256": file_sha256(third_target_path),
            "provider_result_sha256": third_target["provider_result"]["sha256"],
            "grasp_relation_sha256": third_target[
                "transport_grasp_relation"
            ]["sha256"],
            "composer_source_sha256": file_sha256(composer_path),
            "capture_id": third_target["capture_id"],
            "assembly_key_pose_consumed": False,
            "online_object_or_semantic_truth_used": False,
        }
        physical_trace = _initial_trace(
            arguments,
            inputs,
            visual_grasp,
            third_motion_plan,
            dynamic_for_grasp,
        )
        accumulated = dict(trace)
        trace.clear()
        trace.update(physical_trace)
        trace.update(accumulated)
        trace.update(
            {
                "schema_version": physical_trace["schema_version"],
                "mode": "grasp-lift",
                "execution_mode": VISION_GRASP_SERVO_MODE,
                "criteria": physical_trace["criteria"],
                "registered_grasp": physical_trace["registered_grasp"],
                "motion_plan": physical_trace["motion_plan"],
                "controller_online_signals": physical_trace[
                    "controller_online_signals"
                ],
                "robustness_scenario": arguments.robustness_scenario_name,
                "robustness_perturbation": arguments.robustness_perturbation,
                "visual_transport_target_consumption": (
                    arguments.visual_transport_target_consumption
                ),
                "visual_servo": servo_record,
                "visual_servo_error_record_sha256": file_sha256(
                    servo_record_path
                ),
                "visual_tracking_stopped_before_finger_contact": True,
                "contact_confirmation_velocity_absolute_max_rad_s": (
                    dynamic_for_grasp[
                        "contact_velocity_absolute_max_rad_s"
                    ]
                ),
                "accepted_preflight_bound": False,
                "offline_task_gate_passed": False,
                "formal_dynamic_pass": False,
                "online_object_or_contact_truth_used": False,
                "truth_audit_data_returned_to_controller": False,
                "truth_audit_read_during_run_for_isolated_recording": True,
                "truth_evaluated_only_after_control_finished": True,
                "object_pose_writes_after_start": 0,
                "postgrasp_disturbance_execution": {
                    "requested": False,
                    "started": False,
                    "completed": False,
                    "failure_reason": None,
                },
                "postcontact_failure_action": postcontact_failure_action,
            }
        )
        trace["evidence_binding"] = _evidence_binding(
            repository,
            arguments,
            inputs,
            visual_grasp,
            scene,
            visual_grasp_services["robot_asset"],
        )
        runtime = {
            "world": world,
            "scene": scene,
            "robot_asset": visual_grasp_services["robot_asset"],
            "auditor": visual_grasp_services["truth_auditor"],
            "robot_data": robot_data,
            "object_parts": object_parts,
            "engine_monitor": visual_grasp_services["engine_monitor"],
            "runtime_resources_path": arguments.runtime_resources_path,
            "capacity_audit_sha256": arguments.runtime_resources_document[
                "capacity_audit_sha256"
            ],
            "registered_grasp": visual_grasp,
            "control_plan": visual_control_plan,
            "robot_model": inputs.robot_model,
            "payload_model": visual_payload_model,
            "postgrasp_disturbance": None,
        }
        trace, evaluation, engine_runtime = _finish_run(
            repository, inputs, runtime, trace, outcome
        )
        evaluation["visual_servo"] = servo_record
        evaluation["wrist_ft"] = auditor.summary()
        evaluation["formal_postgrasp_main_key_observed"] = False
        evaluation["same_reset_postgrasp_key_observation_executed"] = False
        evaluation["postcontact_failure_action"] = postcontact_failure_action
        evaluation["visual_servo_grasp_physical_pass"] = bool(
            servo_record["position_error_decreased"] is True
            and evaluation.get("nominal_diagnostic_pass") is True
            and auditor.first_ft_stop is None
        )
        return trace, evaluation, engine_runtime

    prefix_document = arguments.visual_second_short_prefix_document
    prefix_requested = prefix_document is not None
    prefix_success: bool | None = None
    prefix_command_count = 0
    free_space_descent_command_count = 0
    prefix_target_path: Path | None = None
    prefix_plan_path: Path | None = None
    prefix_static_path: Path | None = None
    prefix_samples_path: Path | None = None
    prefix_execution_path: Path | None = None
    prefix_execution: dict[str, Any] | None = None
    if prefix_requested and second_ready:
        prefix_contract = prefix_document["short_prefix"]
        gate_contract = prefix_document["runtime_static_gate"]
        second_target = build_visual_transport_target(
            provider_result_path=second_result_path,
            relation_path=transport_relation_path,
            repository_root=repository,
        )
        prefix_target_path = reobserve_root / "second_visual_transport_target.json"
        _write_json_sealed(prefix_target_path, second_target)
        loaded_second_target, _ = load_visual_transport_target(
            prefix_target_path, repository
        )
        if loaded_second_target != second_target:
            raise RuntimeError("second visual target changed during validation")
        second_world_from_object = np.asarray(
            second_target["world_from_transport_object_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        second_motion_plan = control.build_joint_motion_plan(
            repository,
            inputs,
            visual_control_plan,
            second_world_from_object,
            include_lift=grasp_servo,
        )
        second_planned_target = np.asarray(
            second_motion_plan["world_from_hand_base_target"],
            dtype=np.float64,
        ).reshape(4, 4)
        second_target_from_file = np.asarray(
            second_target["world_from_hand_base_target_row_major"],
            dtype=np.float64,
        ).reshape(4, 4)
        if not np.allclose(
            second_planned_target,
            second_target_from_file,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("second motion plan did not consume second RGB-D target")
        second_approach = np.asarray(
            second_motion_plan["approach_arm_waypoints_rad"], dtype=np.float64
        )
        if second_approach.ndim != 2 or second_approach.shape[1] != 7:
            raise RuntimeError("second visual short-prefix waypoints are unavailable")
        frozen_hand = np.asarray(
            second_motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
        )
        if not np.allclose(
            frozen_hand,
            np.asarray(motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("second visual plan changed the frozen hand opening")
        target_method = str(
            prefix_contract.get("target_method", "PLANNER_WAYPOINT_INDEX")
        )
        waypoint_index: int | None = None
        exact_ik_record: dict[str, object] | None = None
        if target_method == "PLANNER_WAYPOINT_INDEX":
            waypoint_index = int(prefix_contract["target_waypoint_index"])
            if not 0 <= waypoint_index < len(second_approach):
                raise RuntimeError("second visual short-prefix waypoint is unavailable")
            prefix_target_arm = second_approach[waypoint_index].copy()
        elif target_method == "BOUNDED_IK_EXACT_CARTESIAN_CLEARANCE":
            seed_indices = tuple(
                int(value) for value in prefix_contract["ik_seed_waypoint_indices"]
            )
            if (
                len(seed_indices) != 2
                or any(not 0 <= index < len(second_approach) for index in seed_indices)
            ):
                raise RuntimeError("exact short-prefix IK seed waypoints are invalid")
            seed = 0.5 * (
                second_approach[seed_indices[0]]
                + second_approach[seed_indices[1]]
            )
            exact_target = second_planned_target.copy()
            exact_target[:3, 3] -= np.asarray(
                second_motion_plan["approach_direction_world"], dtype=np.float64
            ) * float(prefix_contract["target_clearance_from_pregrasp_m"])
            prefix_target_arm, ik_position_error, ik_rotation_error, ik_seed = (
                control.solve_bounded_hand_base_ik(
                    inputs.config.section("ik")["solver"],
                    model=inputs.robot_model,
                    hand_positions=frozen_hand,
                    target_world_from_hand_base=exact_target,
                    seed_arm_positions=(seed,),
                    label="TE_SECOND_VISUAL_EXACT_SHORT_PREFIX",
                )
            )
            prefix_target_arm = np.asarray(prefix_target_arm, dtype=np.float64)
            exact_ik_record = {
                "method": target_method,
                "seed_waypoint_indices": list(seed_indices),
                "seed_arm_rad": seed.tolist(),
                "position_error_m": float(ik_position_error),
                "rotation_error_rad": float(ik_rotation_error),
                "seed_index": int(ik_seed),
            }
        else:
            raise RuntimeError("unsupported second visual short-prefix target method")
        if prefix_target_arm.shape != (7,) or not np.all(
            np.isfinite(prefix_target_arm)
        ):
            raise RuntimeError("second visual short-prefix arm target is invalid")
        prefix_plan_record = {
            "schema_version": "kcg_te_visual_second_short_prefix_plan_v1",
            "capture_id": second_capture_id,
            "second_provider_result_sha256": file_sha256(second_result_path),
            "second_visual_target_sha256": file_sha256(prefix_target_path),
            "second_provider_and_target_sealed_before_plan": True,
            "second_motion_plan": _json_ready(second_motion_plan),
            "target_method": target_method,
            "selected_waypoint_index": waypoint_index,
            "selected_arm_target_rad": prefix_target_arm.tolist(),
            "exact_ik": exact_ik_record,
            "target_clearance_from_pregrasp_m": float(
                prefix_contract["target_clearance_from_pregrasp_m"]
            ),
            "finger_closure_command_count": 0,
            "object_contact_authorized": False,
            "online_object_semantic_instance_or_contact_truth_used": False,
        }
        prefix_plan_path = reobserve_root / "second_visual_short_prefix_plan.json"
        _write_json_sealed(prefix_plan_path, prefix_plan_record)

        current_all = _host_array(robot.get_dof_positions(indices=0))[0]
        current_active = np.asarray(current_all[active_indices], dtype=np.float64)
        if current_active.shape != (11,) or not np.all(np.isfinite(current_active)):
            raise RuntimeError("active joint readback is unavailable before short prefix")
        prefix_static = _audit_visual_second_short_prefix(
            repository=repository,
            inputs=inputs,
            first_planned_high_arm=np.asarray(
                motion_plan["approach_arm_waypoints_rad"], dtype=np.float64
            )[0],
            actual_start_arm=current_active[:7],
            actual_start_hand=current_active[7:],
            second_target_arm=prefix_target_arm,
            frozen_hand_positions=frozen_hand,
            second_pregrasp_world_from_hand=second_planned_target,
            approach_direction_world=second_motion_plan[
                "approach_direction_world"
            ],
            high_panel=high["static_path_panel"],
            prefix_contract=prefix_contract,
            gate_contract=gate_contract,
            world_from_transport_object=second_world_from_object,
        )
        prefix_static.update(
            {
                "first_motion_plan_sha256": file_sha256(motion_plan_path),
                "second_motion_plan_sha256": file_sha256(prefix_plan_path),
                "second_visual_target_sha256": file_sha256(prefix_target_path),
                "runtime_audit_sealed_before_any_prefix_command": True,
            }
        )
        prefix_static_path = reobserve_root / "second_visual_short_prefix_static_audit.json"
        _write_json_sealed(prefix_static_path, prefix_static)
        if prefix_static["pass"] is not True:
            final_online_calls = int(
                command_api_counter.get("set_dof_position_targets", 0)
                - initialization_api_counts.get("set_dof_position_targets", 0)
            )
            trace.update(
                {
                    "high_wait_reached": True,
                    "second_provider_ready": True,
                    "second_visual_target_constructed": True,
                    "second_visual_short_prefix_static_gate_pass": False,
                    "second_visual_short_path_command_count": 0,
                    "free_space_descent_command_count": 0,
                    "descent_to_pregrasp_or_contact_command_count": 0,
                    "finger_closure_command_count": 0,
                    "robot_motion_command_count": final_online_calls,
                    "online_object_or_contact_truth_used": False,
                    "truth_audit_data_returned_to_controller": False,
                    "formal_dynamic_pass": False,
                    "control_authorized": False,
                }
            )
            _write_json_sealed(output / "trace.json", trace)
            failure = {
                "schema_version": "kcg_te_visual_second_short_prefix_result_v1",
                "status": "IMPLEMENTING",
                "physical_result": (
                    "ROBOT_REACHED_SECOND_RGBD_50MM_HIGH_BUT_STATIC_PREFIX_REJECTED"
                ),
                "high_wait_reached": True,
                "second_provider_ready": True,
                "second_visual_short_prefix_static_gate_pass": False,
                "second_visual_short_path_command_count": 0,
                "free_space_descent_command_count": 0,
                "descent_to_pregrasp_or_contact_command_count": 0,
                "finger_closure_command_count": 0,
                "truth_read": False,
                "hardware_authorized": False,
                "evidence": {
                    "second_provider_sha256": file_sha256(second_result_path),
                    "second_target_sha256": file_sha256(prefix_target_path),
                    "second_plan_sha256": file_sha256(prefix_plan_path),
                    "static_audit_sha256": file_sha256(prefix_static_path),
                },
            }
            _write_json_sealed(output / "result.json", failure)
            print(json.dumps(failure, ensure_ascii=False, indent=2))
            return 2

        prefix_command_counts_before = dict(command_api_counter)
        prefix_sample_start = len(auditor.samples)
        prefix_step_start = stepper.step_index
        start_arm = current_active[:7].copy()
        target_arm = prefix_target_arm.copy()
        motion_steps = round(float(prefix_contract["motion_duration_s"]) / dt)
        if motion_steps != int(prefix_contract["motion_command_count"]):
            raise RuntimeError("second-prefix motion command count differs")
        completed_motion_steps = 0
        for index in stepper.active_steps(motion_steps):
            blend = control.minimum_jerk_blend((index + 1) / motion_steps)
            arm_target = (1.0 - blend) * start_arm + blend * target_arm
            stepper.advance(
                str(prefix_contract["motion_phase"]), arm_target, frozen_hand
            )
            completed_motion_steps += 1
        settled = False
        settled_count = 0
        final_arm_error: float | None = None
        completed_settle_steps = 0
        settle_steps = round(float(prefix_contract["settle_timeout_s"]) / dt)
        if stepper.abort_reason is None and completed_motion_steps == motion_steps:
            for _ in stepper.active_steps(settle_steps):
                latest = stepper.advance(
                    str(prefix_contract["settle_phase"]), target_arm, frozen_hand
                )
                completed_settle_steps += 1
                if latest is None:
                    break
                final_arm_error = float(
                    np.max(np.abs(np.asarray(latest[0][:7]) - target_arm))
                )
                settled_count = (
                    settled_count + 1
                    if final_arm_error <= float(prefix_contract["settle_error_rad"])
                    else 0
                )
                if settled_count >= int(
                    prefix_contract["settle_consecutive_samples"]
                ):
                    settled = True
                    break
        if (
            stepper.abort_reason is None
            and completed_motion_steps == motion_steps
            and not settled
        ):
            stepper.abort_reason = "SECOND_VISUAL_SHORT_PATH_NOT_SETTLED"
        prefix_success = bool(
            stepper.abort_reason is None
            and completed_motion_steps == motion_steps
            and settled
        )
        prefix_command_count = int(
            command_api_counter.get("set_dof_position_targets", 0)
            - prefix_command_counts_before.get("set_dof_position_targets", 0)
        )
        prefix_samples = auditor.samples[prefix_sample_start:]
        prefix_samples_path = reobserve_root / "second_visual_short_prefix_joint_ft_samples.json"
        _write_json_sealed(
            prefix_samples_path,
            {
                "schema_version": "kcg_te_visual_second_short_prefix_samples_v1",
                "capture_id": second_capture_id,
                "physical_ft_source": "hand2arm",
                "sample_count": len(prefix_samples),
                "samples": prefix_samples,
                "online_object_semantic_instance_or_contact_truth_used": False,
            },
        )
        final_active = (
            current_active
            if stepper.latest is None
            else np.asarray(stepper.latest[0], dtype=np.float64)
        )
        final_arm_error = float(
            np.max(np.abs(final_active[:7] - target_arm))
        )
        final_hand_transform = np.asarray(
            inputs.robot_model.forward_kinematics(
                tuple(final_active), enforce_limits=False
            )["handbase_link"],
            dtype=np.float64,
        )
        target_hand_transform = np.asarray(
            inputs.robot_model.forward_kinematics(
                tuple(np.concatenate((target_arm, frozen_hand))),
                enforce_limits=False,
            )["handbase_link"],
            dtype=np.float64,
        )
        prefix_phase_counts = {
            phase: int(auditor.phase_counts.get(phase, 0))
            for phase in (
                str(prefix_contract["motion_phase"]),
                str(prefix_contract["settle_phase"]),
            )
        }
        free_space_descent_command_count = int(
            prefix_phase_counts[str(prefix_contract["motion_phase"])]
        )
        prefix_execution = {
            "schema_version": "kcg_te_visual_second_short_prefix_execution_v1",
            "capture_id": second_capture_id,
            "second_provider_sha256": file_sha256(second_result_path),
            "second_target_sha256": file_sha256(prefix_target_path),
            "second_plan_sha256": file_sha256(prefix_plan_path),
            "static_audit_sha256": file_sha256(prefix_static_path),
            "joint_ft_samples_sha256": file_sha256(prefix_samples_path),
            "control_trajectory_sealed_before_truth_read": True,
            "prefix_step_start": prefix_step_start,
            "motion_steps_requested": motion_steps,
            "motion_steps_completed": completed_motion_steps,
            "settle_steps_completed": completed_settle_steps,
            "phase_counts": prefix_phase_counts,
            "position_target_command_count": prefix_command_count,
            "free_space_descent_command_count": (
                free_space_descent_command_count
            ),
            "actual_start_active_joint_positions_rad": current_active.tolist(),
            "target_arm_joint_positions_rad": target_arm.tolist(),
            "frozen_hand_joint_positions_rad": frozen_hand.tolist(),
            "final_active_joint_positions_rad": final_active.tolist(),
            "final_maximum_arm_joint_error_rad": final_arm_error,
            "final_handbase_position_error_m": float(
                np.linalg.norm(
                    final_hand_transform[:3, 3] - target_hand_transform[:3, 3]
                )
            ),
            "final_handbase_rotation_error_rad": _rotation_error_rad(
                final_hand_transform[:3, :3], target_hand_transform[:3, :3]
            ),
            "settled": settled,
            "abort_reason": stepper.abort_reason,
            "success": prefix_success,
            "wrist_ft": auditor.summary(),
            "finger_closure_command_count": 0,
            "descent_to_pregrasp_or_contact_command_count": 0,
            "object_contact_authorized": False,
            "online_object_semantic_instance_or_contact_truth_used": False,
            "hardware_authorized": False,
        }
        prefix_execution_path = reobserve_root / "second_visual_short_prefix_execution.json"
        _write_json_sealed(prefix_execution_path, prefix_execution)

    # Only now, after both providers and any authorized short-prefix control
    # records are sealed, may the evaluator read object truth.  No command or
    # physics step follows this read.
    truth_position, truth_quaternion = object_parts[0].get_world_pose()
    truth_position = _host_array(truth_position).reshape(3)
    truth_quaternion = _host_array(truth_quaternion).reshape(4)
    truth_axis = _quaternion_wxyz_rotation(truth_quaternion) @ np.asarray(
        (0.0, 0.0, 1.0), dtype=np.float64
    )
    second_position = (
        np.asarray(second_transport["position_xyz_m"], dtype=np.float64)
        if second_ready
        else None
    )
    second_axis = (
        np.asarray(second_transport["outward_axis_world"], dtype=np.float64)
        if second_ready
        else None
    )
    posthoc_axis_error = (
        float(
            math.acos(
                np.clip(
                    float(second_axis @ truth_axis)
                    / float(
                        np.linalg.norm(second_axis) * np.linalg.norm(truth_axis)
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        if second_ready
        else None
    )
    workflow_success = bool(
        second_ready
        and (not prefix_requested or prefix_success is True)
    )
    posthoc = {
        "schema_version": "kcg_te_visual_high_reobserve_posthoc_truth_v1",
        "capture_id": second_capture_id,
        "reset_uuid": reset_uuid,
        "first_provider_sha256": file_sha256(provider_result_path),
        "second_provider_sha256": file_sha256(second_result_path),
        "both_provider_results_sealed_before_truth_read": True,
        "truth_read_count": 1,
        "truth_returned_to_provider_or_controller": False,
        "posthoc_truth_position_xyz_m": truth_position.tolist(),
        "posthoc_truth_orientation_wxyz": truth_quaternion.tolist(),
        "second_estimated_position_xyz_m": (
            None if second_position is None else second_position.tolist()
        ),
        "second_signed_position_error_xyz_m": (
            None
            if second_position is None
            else (second_position - truth_position).tolist()
        ),
        "second_estimated_outward_axis_world": (
            None if second_axis is None else second_axis.tolist()
        ),
        "posthoc_truth_outward_axis_world": truth_axis.tolist(),
        "second_axis_tilt_error_rad": posthoc_axis_error,
        "second_provider_ready": second_ready,
        "second_visual_short_prefix_requested": prefix_requested,
        "second_visual_short_prefix_success": prefix_success,
        "second_visual_short_path_command_count": prefix_command_count,
        "free_space_descent_command_count": free_space_descent_command_count,
        "descent_to_pregrasp_or_contact_command_count": 0,
        "short_prefix_execution_sealed_before_truth_read": (
            None
            if prefix_execution is None
            else prefix_execution.get(
                "control_trajectory_sealed_before_truth_read"
            )
        ),
        "robot_commands_after_truth_read": 0,
        "descent_contact_or_closure_commands": 0,
        "evidence_limit": (
            "ONE_SIMULATION_ONLY_HIGH_OBSERVATION_AND_OPTIONAL_FREE_SPACE_"
            "PREFIX_RUN_NOT_GRASP_CONTACT_OR_ASSEMBLY_SUCCESS"
        ),
    }
    posthoc_path = reobserve_root / "posthoc_truth_evaluation.json"
    _write_json_sealed(posthoc_path, posthoc)
    trace.update(
        {
            "provider_result_sha256": file_sha256(provider_result_path),
            "provider_result_sealed_before_truth_read": True,
            "motion_plan_constructed": True,
            "world_pregrasp_target_consumed": True,
            "fresh_visual_target_sha256": file_sha256(target_path),
            "fresh_motion_plan_sha256": file_sha256(motion_plan_path),
            "high_motion_execution_sha256": file_sha256(high_motion_path),
            "high_motion_samples_sha256": file_sha256(samples_path),
            "second_provider_result_sha256": file_sha256(second_result_path),
            "second_provider_sealed_before_truth_read": True,
            "sealed_reobservation_report_sha256": file_sha256(
                reobserve_report_path
            ),
            "posthoc_truth_evaluation_sha256": file_sha256(posthoc_path),
            "robot_target_position_velocity_api_call_count": sum(
                command_api_counter.values()
            ),
            "robot_motion_command_count": int(
                command_api_counter.get("set_dof_position_targets", 0)
                - initialization_api_counts.get("set_dof_position_targets", 0)
            ),
            "second_visual_short_path_command_count": prefix_command_count,
            "finger_closure_command_count": 0,
            "online_object_or_contact_truth_used": False,
            "truth_audit_data_returned_to_controller": False,
            "formal_dynamic_pass": False,
            "control_authorized": False,
            "high_wait_reached": high_reached,
            "second_provider_ready": second_ready,
            "second_visual_short_prefix_requested": prefix_requested,
            "second_visual_short_prefix_success": prefix_success,
            "second_visual_target_sha256": (
                None
                if prefix_target_path is None
                else file_sha256(prefix_target_path)
            ),
            "second_visual_motion_plan_sha256": (
                None if prefix_plan_path is None else file_sha256(prefix_plan_path)
            ),
            "second_visual_static_audit_sha256": (
                None
                if prefix_static_path is None
                else file_sha256(prefix_static_path)
            ),
            "second_visual_execution_sha256": (
                None
                if prefix_execution_path is None
                else file_sha256(prefix_execution_path)
            ),
        }
    )
    if prefix_requested:
        trace.pop("descent_command_count", None)
        trace["free_space_descent_command_count"] = (
            free_space_descent_command_count
        )
        trace["descent_to_pregrasp_or_contact_command_count"] = 0
    else:
        trace["descent_command_count"] = 0
    _write_json_sealed(output / "trace.json", trace)
    result = {
        "schema_version": (
            "kcg_te_visual_second_short_prefix_result_v1"
            if prefix_requested
            else "kcg_te_visual_high_reobserve_result_v1"
        ),
        "status": "DYNAMIC_PASS" if workflow_success else "IMPLEMENTING",
        "physical_result": (
            "ROBOT_REACHED_SECOND_RGBD_DRIVEN_FREE_SPACE_PREFIX"
            if prefix_requested and prefix_success is True
            else (
                "ROBOT_REACHED_SECOND_RGBD_HIGH_BUT_SHORT_PREFIX_STOPPED"
                if prefix_requested and second_ready
                else (
                    "ROBOT_REACHED_FRESH_VISUAL_50MM_APPROACH_HIGH_AND_REOBSERVED"
                    if second_ready
                    else "ROBOT_REACHED_FRESH_VISUAL_50MM_APPROACH_HIGH_BUT_REOBSERVATION_MISS"
                )
            )
        ),
        "high_wait_reached": high_reached,
        "second_provider_ready": second_ready,
        "second_visual_short_path_command_count": prefix_command_count,
        "second_visual_short_prefix_success": prefix_success,
        "target_clearance_from_pregrasp_m": (
            None
            if not prefix_requested
            else float(
                prefix_document["short_prefix"][
                    "target_clearance_from_pregrasp_m"
                ]
            )
        ),
        "target_method": (
            None
            if not prefix_requested
            else prefix_document["short_prefix"].get(
                "target_method", "PLANNER_WAYPOINT_INDEX"
            )
        ),
        "finger_closure_command_count": 0,
        "descent_contact_grasp_insertion_or_locking_occurred": False,
        "formal_vision_driven_grasp_claimed": False,
        "hardware_authorized": False,
        "evidence": {
            "high_motion_execution_sha256": file_sha256(high_motion_path),
            "sealed_reobservation_report_sha256": file_sha256(
                reobserve_report_path
            ),
            "posthoc_truth_evaluation_sha256": file_sha256(posthoc_path),
            "second_visual_target_sha256": (
                None
                if prefix_target_path is None
                else file_sha256(prefix_target_path)
            ),
            "second_visual_motion_plan_sha256": (
                None if prefix_plan_path is None else file_sha256(prefix_plan_path)
            ),
            "second_visual_static_audit_sha256": (
                None
                if prefix_static_path is None
                else file_sha256(prefix_static_path)
            ),
            "second_visual_execution_sha256": (
                None
                if prefix_execution_path is None
                else file_sha256(prefix_execution_path)
            ),
        },
    }
    if prefix_requested:
        result["free_space_descent_command_count"] = (
            free_space_descent_command_count
        )
        result["descent_to_pregrasp_or_contact_command_count"] = 0
    else:
        result["descent_command_count"] = 0
    _write_json_sealed(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if workflow_success else 2


def _create_runtime(
    repository, arguments, inputs, grasp, scene_entry, motion_plan, trace,
    simulation_app, output,
):
    import carb.settings
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from pxr import (
        Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdLux, UsdPhysics,
        UsdShade,
    )

    dynamic = arguments.dynamic_settings
    robot_asset = arguments.robot_asset_path
    if not robot_asset.is_file():
        raise ValueError("selected robot asset is missing")
    trace["robot_asset"] = str(robot_asset)
    trace["robot_asset_override_used"] = arguments.robot_asset_override_used
    settings = carb.settings.get_settings()
    contact_processing_before = settings.get(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    contact_processing_before_world = settings.get_as_bool(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    trace["contact_processing_setting_audit"] = {
        "path": SETTING_DISABLE_CONTACT_PROCESSING,
        "before": contact_processing_before,
        "required": False,
        "before_world": contact_processing_before_world,
    }
    if contact_processing_before_world:
        raise RuntimeError("PhysX contact processing was not enabled before World creation")
    World.clear_instance()
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=float(dynamic["physics_dt_s"]),
        rendering_dt=1.0 / 60.0,
        **gpu_world_parameters(arguments.runtime_resources_document),
    )
    context = world.get_physics_context()
    stage = get_current_stage()
    physics_scene_prim = stage.GetPrimAtPath(context.prim_path)
    physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    minimum_velocity_iterations = (
        physics_scene_api.GetMinVelocityIterationCountAttr().Get()
    )
    physics_scene_api.CreateMinVelocityIterationCountAttr().Set(8)
    observed_minimum_velocity_iterations = (
        physics_scene_api.GetMinVelocityIterationCountAttr().Get()
    )
    if observed_minimum_velocity_iterations != 8:
        raise RuntimeError("physics scene minimum velocity iterations did not read back as 8")
    trace["physics_scene_velocity_iteration_audit"] = {
        "before": minimum_velocity_iterations,
        "required": 8,
        "observed": observed_minimum_velocity_iterations,
    }
    scene = prepare_dynamic_scene(
        repository,
        stage,
        scene_entry,
        add_reference_to_stage,
        arguments.robustness_perturbation,
    )
    add_reference_to_stage(str(robot_asset), ROBOT_ROOT)
    _apply_contact_friction_perturbation(
        stage, scene, Usd, UsdPhysics, UsdShade, PhysxSchema
    )
    _apply_object_mass_perturbation(stage, scene, Gf, UsdPhysics)
    _apply_center_of_mass_perturbation(stage, scene, Gf, UsdGeom, UsdPhysics)
    center_delta = (
        np.zeros(3, dtype=np.float64)
        if scene["requested_center_of_mass_delta_object_m"] is None
        else np.asarray(
            scene["requested_center_of_mass_delta_object_m"], dtype=np.float64
        )
    )
    object_from_hand = np.asarray(
        grasp["control_plan"]["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    hand_from_object = np.linalg.inv(object_from_hand)
    center_object = (
        np.asarray(inputs.object_contract.model.center_of_mass_m, dtype=np.float64)
        + center_delta
    )
    center_hand = (hand_from_object @ np.concatenate((center_object, (1.0,))))[:3]
    payload_model = {
        "method": "CAD_MASS_COM_JACOBIAN_FEEDFORWARD_RAMPED_OVER_TABLE_CLEARANCE",
        "mass_kg": (
            float(inputs.object_contract.model.mass_kg)
            * float(scene["object_mass_audit"]["effective_scale"])
        ),
        "gravity_m_s2": abs(float(scene["gravity_m_s2"])),
        "center_of_mass_object_m": center_object.tolist(),
        "center_of_mass_from_hand_m": center_hand.tolist(),
        "transfer_distance_m": float(dynamic["table_release_clearance_m"]),
        "online_object_truth_used": False,
    }
    if arguments.capture_visual_evidence:
        render = scene["render"]
        lighting_root = "/World/CARTSGraspVisualEvidenceLights"
        dome = UsdLux.DomeLight.Define(stage, lighting_root + "/Fill")
        dome.CreateIntensityAttr(float(render.dome_light_intensity))
        dome.CreateColorAttr(Gf.Vec3f(*render.dome_light_color_rgb))
        key = UsdLux.DistantLight.Define(stage, lighting_root + "/Key")
        key.CreateIntensityAttr(float(render.key_light_intensity))
        key.CreateColorAttr(Gf.Vec3f(*render.key_light_color_rgb))
        key.AddRotateXYZOp().Set(
            Gf.Vec3f(*render.key_light_rotation_degrees_xyz)
        )
    trace["evidence_binding"] = _evidence_binding(
        repository, arguments, inputs, grasp, scene, robot_asset)
    if arguments.mode in ("first-finger-diagnostic", "grasp-lift"):
        preflight = arguments.preflight_document
        preflight_binding = preflight.get("evidence_binding", {})
        current_binding = trace["evidence_binding"]
        # Preflight stops at the open-hand approach pose, so clamp effort is
        # neither commanded nor able to affect its geometry/collision result.
        # Permit one accepted geometric preflight to serve a controlled effort
        # sweep while every physical binding that can affect preflight remains
        # identical.
        clamp_only_binding_keys = {
            "registered_grasp_sha256",
            "required_closing_joint_effort_nm",
            "predicted_unit_task_closing_joint_effort_nm",
        }
        compared_binding_keys = (
            set(preflight_binding) | set(current_binding)
        ) - clamp_only_binding_keys
        mismatched_binding_keys = sorted(
            key
            for key in compared_binding_keys
            if preflight_binding.get(key) != current_binding.get(key)
        )
        if mismatched_binding_keys:
            raise ValueError(
                "preflight evidence binding does not match this run outside "
                f"clamp-only fields: {mismatched_binding_keys}"
            )
        trace["preflight_clamp_effort_reuse_audit"] = {
            "preflight_required_closing_joint_effort_nm": (
                preflight_binding.get("required_closing_joint_effort_nm")
            ),
            "current_required_closing_joint_effort_nm": (
                current_binding.get("required_closing_joint_effort_nm")
            ),
            "ignored_binding_keys": sorted(clamp_only_binding_keys),
            "all_other_binding_keys_match": True,
        }
        trace["accepted_preflight_bound"] = True
        trace["accepted_preflight_evaluation_sha256"] = file_sha256(
            arguments.preflight_evaluation_path)
    rigid_body_prims, contact_report_prims = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_prims.append(str(prim.GetPath()))
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            contact_report_prims.append(str(prim.GetPath()))
    trace["contact_report_api_audit"] = {
        "before_reset_rigid_body_paths": rigid_body_prims,
        "before_reset_reporter_paths": contact_report_prims,
    }
    hand_base_prim = stage.GetPrimAtPath(HAND_BASE_PATH)
    if not hand_base_prim.IsValid():
        raise RuntimeError("hand base prim is missing")
    object_parts = tuple(
        world.scene.add(
            SingleRigidPrim(prim_path=path, name=f"carts_v2_object_part_{index}")
        )
        for index, path in enumerate(scene["part_prim_paths"])
    )
    ft_articulation = (
        world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name="te_visual_high_hand2arm_reaction_reader",
            )
        )
        if arguments.mode in VISION_MOTION_MODES
        else None
    )
    robot_contact_paths = tuple(
        path for path in rigid_body_prims
        if path == ROBOT_ROOT or path.startswith(ROBOT_ROOT + "/")
    )
    object_contact_paths = tuple(map(str, scene["part_prim_paths"]))
    tensor_contact_sensor_paths = robot_contact_paths + object_contact_paths
    if (
        len(set(tensor_contact_sensor_paths)) != len(tensor_contact_sensor_paths)
        or not set(object_contact_paths).issubset(rigid_body_prims)
    ):
        raise RuntimeError("tensor contact sensor paths do not match audited rigid bodies")
    tensor_contact_prim = TensorRigidPrim(
        list(tensor_contact_sensor_paths),
        resolve_paths=False,
        contact_filter_paths=list(object_contact_paths),
        max_contact_count=TENSOR_CONTACT_MAX_COUNT,
    )
    trace["tensor_contact_view_audit"] = {
        "robot_sensor_paths": list(robot_contact_paths),
        "object_sensor_paths": list(object_contact_paths),
        "contact_filter_paths": list(object_contact_paths),
        "sensor_paths": list(tensor_contact_sensor_paths),
        "max_contact_count": TENSOR_CONTACT_MAX_COUNT,
    }
    context.set_gravity(float(scene["gravity_m_s2"]))
    world.reset()
    if arguments.postgrasp_disturbance is not None:
        retain_rows = []
        for path in scene["part_prim_paths"]:
            value = PhysxSchema.PhysxRigidBodyAPI(
                stage.GetPrimAtPath(path)
            ).GetRetainAccelerationsAttr().Get()
            retain_rows.append({
                "rigid_prim_path": str(path),
                "retain_accelerations": bool(value) if value is not None else False,
                "attribute_authored": value is not None,
            })
        if any(row["retain_accelerations"] for row in retain_rows):
            raise RuntimeError(
                "postgrasp external wrench requires per-step force clearing"
            )
        trace["postgrasp_disturbance"]["retain_accelerations_audit"] = (
            retain_rows
        )
    tensor_contact_view_valid = (
        tensor_contact_prim.is_physics_tensor_entity_valid()
    )
    trace["tensor_contact_view_audit"]["valid_after_reset"] = (
        tensor_contact_view_valid
    )
    if not tensor_contact_view_valid:
        raise RuntimeError("tensor contact view is invalid after reset")
    contact_processing_after_reset = settings.get_as_bool(
        SETTING_DISABLE_CONTACT_PROCESSING
    )
    trace["contact_processing_setting_audit"]["after_reset"] = (
        contact_processing_after_reset
    )
    if contact_processing_after_reset:
        raise RuntimeError("PhysX contact processing was disabled during reset")
    after_rigid = [str(prim.GetPath()) for prim in stage.Traverse()
                   if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    after_reporters = [str(prim.GetPath()) for prim in stage.Traverse()
                       if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)]
    trace["contact_report_api_audit"].update({
        "after_reset_rigid_body_paths": after_rigid,
        "after_reset_reporter_paths": after_reporters,
        "complete": (set(rigid_body_prims) == set(contact_report_prims)
                     == set(after_rigid) == set(after_reporters)),
    })
    backend = gpu_backend_record(world, context)
    if not backend["pass"]:
        raise RuntimeError(f"GPU physics backend audit failed: {backend}")
    trace["physics_backend"] = backend
    visual_grasp_services = None
    if arguments.mode == VISION_GRASP_SERVO_MODE:
        visual_engine_monitor = PhysxStatsMonitor(context)
        visual_truth_auditor = TruthAuditRecorder(
            object_parts=object_parts,
            hand_base_prim=hand_base_prim,
            robot_model=inputs.robot_model,
            stage_modules=(Gf, Usd, UsdGeom),
            contact_interface=get_physx_simulation_interface(),
            path_decoder=PhysicsSchemaTools.intToSdfPath,
            roots={"robot": ROBOT_ROOT, **scene["roots"]},
            expected_total_mass_kg=(
                inputs.object_contract.model.mass_kg
                * scene["object_mass_audit"]["effective_scale"]
            ),
            part_bottom_offsets_m=scene["part_bottom_offsets_m"],
            table_top_z_m=scene["table_top_z_m"],
            physics_dt_s=float(dynamic["physics_dt_s"]),
            engine_monitor=visual_engine_monitor,
            physics_step_interface=get_physx_interface(),
            tensor_contact_prim=tensor_contact_prim,
            tensor_contact_sensor_paths=tensor_contact_sensor_paths,
            tensor_contact_max_count=TENSOR_CONTACT_MAX_COUNT,
        )
        visual_grasp_services = {
            "truth_auditor": visual_truth_auditor,
            "engine_monitor": visual_engine_monitor,
            "payload_model": payload_model,
            "robot_asset": robot_asset,
        }
    if arguments.mode in (SAME_RESET_RGBD_MODE, *VISION_MOTION_MODES):
        visual_result = _run_same_reset_rgbd_observe(
            repository,
            arguments,
            inputs,
            scene_entry,
            world,
            stage,
            scene,
            object_parts,
            grasp,
            ft_articulation,
            simulation_app,
            output,
            trace,
            visual_grasp_services=visual_grasp_services,
        )
        return {
            "visual_grasp_result": visual_result
            if arguments.mode == VISION_GRASP_SERVO_MODE
            else None,
            "same_reset_rgbd_exit_code": visual_result
            if arguments.mode != VISION_GRASP_SERVO_MODE
            else None,
        }
    engine_monitor = PhysxStatsMonitor(context)
    robot_data = control.create_native_gravity_compensated_robot(
        ARTICULATION_PATH,
        EXPECTED_DOF_NAMES,
        dynamic,
        initial_arm_positions=(
            motion_plan["pregrasp_arm_positions_rad"]
            if arguments.initialize_at_pregrasp
            else None
        ),
        initial_hand_positions=(
            motion_plan["pregrasp_hand_positions_rad"]
            if arguments.initialize_at_pregrasp
            else None
        ),
    )
    trace["initial_joint_audit"] = audit_initial_joint_state(
        robot_data[0], robot_data[0].dof_names
    )
    auditor = TruthAuditRecorder(
        object_parts=object_parts,
        hand_base_prim=hand_base_prim,
        robot_model=inputs.robot_model,
        stage_modules=(Gf, Usd, UsdGeom),
        contact_interface=get_physx_simulation_interface(),
        path_decoder=PhysicsSchemaTools.intToSdfPath,
        roots={"robot": ROBOT_ROOT, **scene["roots"]},
        expected_total_mass_kg=(
            inputs.object_contract.model.mass_kg
            * scene["object_mass_audit"]["effective_scale"]
        ),
        part_bottom_offsets_m=scene["part_bottom_offsets_m"],
        table_top_z_m=scene["table_top_z_m"],
        physics_dt_s=float(dynamic["physics_dt_s"]),
        engine_monitor=engine_monitor,
        physics_step_interface=get_physx_interface(),
        tensor_contact_prim=tensor_contact_prim,
        tensor_contact_sensor_paths=tensor_contact_sensor_paths,
        tensor_contact_max_count=TENSOR_CONTACT_MAX_COUNT,
    )
    return {
        "world": world, "scene": scene, "robot_asset": robot_asset,
        "auditor": auditor, "robot_data": robot_data,
        "object_parts": object_parts,
        "engine_monitor": engine_monitor,
        "runtime_resources_path": arguments.runtime_resources_path,
        "capacity_audit_sha256": arguments.runtime_resources_document[
            "capacity_audit_sha256"],
        "registered_grasp": grasp,
        "control_plan": grasp["control_plan"],
        "robot_model": inputs.robot_model,
        "payload_model": payload_model,
        "postgrasp_disturbance": arguments.postgrasp_disturbance,
    }


class _VisualEvidenceCapture:
    """Capture post-step viewport frames; never return image truth to control."""

    def __init__(self, *, world, auditor, output: Path, physics_dt_s: float) -> None:
        import omni.kit.renderer_capture
        from omni.kit.viewport.utility import get_active_viewport

        self.world = world
        self.auditor = auditor
        self.output = output / "visuals"
        self.output.mkdir(parents=True, exist_ok=False)
        self.physics_dt_s = float(physics_dt_s)
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("visual evidence requested but no Isaac viewport is active")
        self.viewport.resolution = (1600, 900)
        self.renderer_capture = (
            omni.kit.renderer_capture.acquire_renderer_capture_interface()
        )
        self.records: list[dict[str, object]] = []
        self.pregrasp_object_z_m: float | None = None
        self._captured: set[str] = set()
        self._truth_capture = auditor.capture
        auditor.capture = self.capture

    def capture(self, **kwargs) -> None:
        self._truth_capture(**kwargs)
        row = self.auditor.samples[-1]
        phase = str(row["phase"])
        if phase == "pregrasp_hold":
            if self.pregrasp_object_z_m is None:
                self.pregrasp_object_z_m = float(row["object_center_m"][2])
            self._capture_once("01_pregrasp", row)
        elif phase == "preload":
            self._capture_once("02_three_finger_clamp", row)
        elif (
            phase == "lift"
            and self.pregrasp_object_z_m is not None
            and float(row["object_center_m"][2]) - self.pregrasp_object_z_m >= 0.020
            and int(row["contacts"]["object_table"]) == 0
        ):
            self._capture_once("03_table_released_20mm", row)

    def capture_run_end(self) -> None:
        if not self.auditor.samples:
            return
        row = self.auditor.samples[-1]
        if row["phase"] == "hold":
            name = "04_final_hold"
        elif row["phase"] == "postgrasp_disturbance_recovery":
            name = "06_disturbance_recovery_end"
        else:
            name = "04_run_end_failure"
        self._capture_once(name, row)

    def _capture_once(self, name: str, row: Mapping[str, object]) -> None:
        if name in self._captured:
            return
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file

        center = np.asarray(row["object_center_m"], dtype=np.float64)
        target = center + np.asarray((0.0, 0.0, 0.015), dtype=np.float64)
        eye = target + np.asarray((0.38, 0.34, 0.25), dtype=np.float64)
        set_camera_view(eye=eye, target=target, viewport_api=self.viewport)
        for _ in range(8):
            self.world.render()
        image_path = self.output / f"{name}.png"
        capture_viewport_to_file(self.viewport, file_path=str(image_path))
        for _ in range(16):
            self.world.render()
        self.renderer_capture.wait_async_capture()
        for _ in range(8):
            if image_path.is_file() and image_path.stat().st_size > 0:
                break
            self.world.render()
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise RuntimeError(f"Isaac viewport capture did not produce {image_path}")
        terminal = row["contacts"]["terminal_link_object"]
        self.records.append({
            "file": str(image_path),
            "step": int(row["step"]),
            "simulation_time_s": float(row["step"]) * self.physics_dt_s,
            "phase": str(row["phase"]),
            "object_center_m": list(map(float, row["object_center_m"])),
            "object_lift_from_pregrasp_m": (
                None if self.pregrasp_object_z_m is None else
                float(row["object_center_m"][2]) - self.pregrasp_object_z_m
            ),
            "object_table_contact_count": int(row["contacts"]["object_table"]),
            "terminal_link_object_contact_counts": list(map(int, terminal)),
            "camera_eye_m": eye.tolist(),
            "camera_target_m": target.tolist(),
        })
        self._captured.add(name)


def _run_postgrasp_disturbance(runtime, arguments, stepper, grasp_result):
    contract = arguments.postgrasp_disturbance
    execution = {
        "requested": contract is not None,
        "started": False,
        "completed": False,
        "failure_reason": None,
    }
    if contract is None:
        return execution
    nominal_failure = grasp_result.get("failure_reason") or stepper.abort_reason
    if nominal_failure is not None:
        execution["failure_reason"] = (
            "NOMINAL_CONTROLLER_DID_NOT_COMPLETE_BEFORE_DISTURBANCE:"
            + str(nominal_failure)
        )
        return execution
    parts = runtime["object_parts"]
    if len(parts) == 1:
        part_index = 0
    elif (
        len(parts) == 2
        and arguments.free_split_object_manifest is not None
    ):
        # The hand already acts on the coupling nut.  External test loads act
        # on the connector body so the nut/body joint remains part of the test.
        part_index = 0
    else:
        raise RuntimeError(
            "postgrasp disturbance target rigid body is ambiguous"
        )
    part = parts[part_index]
    target_prim_path = runtime["scene"]["part_prim_paths"][part_index]
    view = getattr(part, "_rigid_prim_view", None)
    if (
        view is None
        or not view.is_physics_handle_valid()
        or int(view.count) != 1
    ):
        raise RuntimeError("postgrasp rigid-body force view is unavailable")

    condition = contract["condition"]
    force_task = np.asarray(condition["force_task_n"], dtype=np.float64)
    moment_task = np.asarray(condition["moment_task_nm"], dtype=np.float64)
    body_com_application = bool(
        contract.get("schema_version") == POSTGRASP_DISTURBANCE_COM_SCHEMA
    )
    if body_com_application:
        frozen_position, frozen_quaternion = part.get_world_pose()
        if hasattr(frozen_position, "detach"):
            frozen_position = frozen_position.detach().cpu()
        if hasattr(frozen_quaternion, "detach"):
            frozen_quaternion = frozen_quaternion.detach().cpu()
        frozen_position = np.asarray(frozen_position, dtype=np.float64)
        frozen_quaternion = np.asarray(frozen_quaternion, dtype=np.float64)
        world_from_object_rotation = _quaternion_wxyz_rotation(
            frozen_quaternion
        )
        coordinate = contract["coordinate_contract"]
        frame_source = coordinate["frozen_world_task_frame_source"]
        if (
            frame_source
            == "CURRENT_RUN_HELD_HAND_POSE_FROZEN_BEFORE_DISTURBANCE"
        ):
            last_sample = runtime["auditor"].samples[-1]
            frozen_hand_position = np.asarray(
                last_sample["hand_base_position_m"], dtype=np.float64
            )
            frozen_hand_quaternion = np.asarray(
                last_sample["hand_base_orientation_wxyz"], dtype=np.float64
            )
            task_from_hand = np.asarray(
                coordinate["task_from_hand_rotation_row_major"],
                dtype=np.float64,
            )
            if (
                frozen_hand_position.shape != (3,)
                or frozen_hand_quaternion.shape != (4,)
                or task_from_hand.shape != (3, 3)
                or not np.all(np.isfinite(frozen_hand_position))
                or not np.all(np.isfinite(frozen_hand_quaternion))
                or not np.all(np.isfinite(task_from_hand))
            ):
                raise RuntimeError(
                    "held-hand disturbance frame is invalid"
                )
            world_from_task = (
                _quaternion_wxyz_rotation(frozen_hand_quaternion)
                @ task_from_hand.T
            )
        elif (
            frame_source
            == "CURRENT_RUN_POSTGRASP_HOLD_BODY_POSE_FROZEN_BEFORE_DISTURBANCE"
        ):
            task_from_object = np.asarray(
                coordinate[
                    "task_from_supplier_object_rotation_row_major"
                ],
                dtype=np.float64,
            )
            world_from_task = (
                world_from_object_rotation @ task_from_object.T
            )
            frozen_hand_position = None
            frozen_hand_quaternion = None
        else:
            raise RuntimeError(
                "unsupported body-COM disturbance vector frame"
            )
        contract["coordinate_contract"][
            "frozen_world_from_task_rotation_row_major"
        ] = world_from_task.tolist()
        contract["coordinate_contract"][
            "frozen_world_task_frame_source_sample"
        ] = {
            "step": int(stepper.step_index),
            "body_origin_world_m": frozen_position.tolist(),
            "body_orientation_world_wxyz": frozen_quaternion.tolist(),
            "held_hand_position_world_m": (
                None
                if frozen_hand_position is None
                else frozen_hand_position.tolist()
            ),
            "held_hand_orientation_world_wxyz": (
                None
                if frozen_hand_quaternion is None
                else frozen_hand_quaternion.tolist()
            ),
            "vector_frame_source": frame_source,
            "used_only_for_test_frame_and_com_application": True,
            "returned_to_grasp_controller": False,
        }
        force_world = world_from_task @ force_task
        moment_world = world_from_task @ moment_task
        condition["force_world_n"] = force_world.tolist()
        condition["moment_world_nm"] = moment_world.tolist()
        application_point_object = np.asarray(
            contract["coordinate_contract"][
                "application_point_supplier_object_m"
            ],
            dtype=np.float64,
        )
    else:
        force_world = np.asarray(condition["force_world_n"], dtype=np.float64)
        moment_world = np.asarray(condition["moment_world_nm"], dtype=np.float64)
        application_point_object = None
    force_cap = float(contract["input_limits"]["maximum_resultant_force_n"])
    moment_cap = float(contract["input_limits"]["maximum_resultant_moment_nm"])
    if (
        not all(np.all(np.isfinite(value)) for value in (
            force_task, moment_task, force_world, moment_world
        ))
        or np.linalg.norm(force_world) > force_cap + 1.0e-12
        or np.linalg.norm(moment_world) > moment_cap + 1.0e-12
    ):
        raise RuntimeError("postgrasp transformed wrench exceeds its cap")

    last = runtime["auditor"].samples[-1]
    active_target = np.asarray(last["active_targets_rad"], dtype=np.float64)
    if active_target.shape != (11,) or not np.all(np.isfinite(active_target)):
        raise RuntimeError("postgrasp held joint target is unavailable")
    arm_target = active_target[:7]
    hand_target = np.array(active_target[7:], copy=True)
    contact_controller = grasp_result.get("contact_controller")
    if contact_controller is None:
        raise RuntimeError("postgrasp contact controller is unavailable")
    closing_direction = np.sign(
        np.asarray(contact_controller.goal, dtype=np.float64)
        - np.asarray(contact_controller.start, dtype=np.float64)
    )[1:]
    if np.any(closing_direction == 0.0):
        raise RuntimeError("postgrasp closing direction is invalid")
    closure_target = np.asarray(
        contact_controller.target, dtype=np.float64
    )
    preload_scales = np.concatenate((
        [0.0],
        np.asarray(
            arguments.dynamic_settings.get(
                "finger_preload_scales", (1.0, 1.0, 1.0)
            ),
            dtype=np.float64,
        ),
    ))
    preload_limit_goal = closure_target + (
        float(arguments.dynamic_settings["preload_increment_rad"])
        * np.sign(
            np.asarray(contact_controller.goal, dtype=np.float64)
            - np.asarray(contact_controller.start, dtype=np.float64)
        )
        * preload_scales
    )
    required_effort = np.asarray(
        arguments.dynamic_settings["required_closing_joint_effort_nm"],
        dtype=np.float64,
    )
    maximum_finger_increment = (
        float(arguments.dynamic_settings["finger_maximum_speed_rad_s"])
        * float(arguments.dynamic_settings["physics_dt_s"])
    )
    hand_stiffness = float(arguments.dynamic_settings["hand_stiffness"])
    if not np.isfinite(hand_stiffness) or hand_stiffness <= 0.0:
        raise RuntimeError("postgrasp hand stiffness is invalid")
    preload_lower = np.minimum(
        np.asarray(contact_controller.start, dtype=np.float64),
        preload_limit_goal,
    )
    preload_upper = np.maximum(
        np.asarray(contact_controller.start, dtype=np.float64),
        preload_limit_goal,
    )
    tare_rows = [
        np.asarray(row["active_efforts_nm"], dtype=np.float64)[7:]
        for row in runtime["auditor"].samples if row["phase"] == "tare"
    ]
    if not tare_rows:
        raise RuntimeError("postgrasp hand effort tare is unavailable")
    tare = np.mean(np.stack(tare_rows), axis=0)
    effort_abort = float(arguments.dynamic_settings["measured_effort_abort_nm"])
    float32_tolerance = 8.0 * np.finfo(np.float32).eps
    submitted_calls = 0
    import torch

    def advance_with_wrench(phase: str, scale: float):
        nonlocal submitted_calls, hand_target
        value = float(scale)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError("postgrasp wrench scale is outside [0, 1]")
        applied_force_task = value * force_task
        applied_moment_task = value * moment_task
        applied_force_world = np.asarray(
            value * force_world, dtype=np.float32
        ).reshape(1, 3)
        applied_moment_world = np.asarray(
            value * moment_world, dtype=np.float32
        ).reshape(1, 3)
        if (
            np.linalg.norm(applied_force_world.astype(np.float64))
            > force_cap * (1.0 + float32_tolerance) + 1.0e-12
            or np.linalg.norm(applied_moment_world.astype(np.float64))
            > moment_cap * (1.0 + float32_tolerance) + 1.0e-12
        ):
            raise RuntimeError("float32 postgrasp wrench exceeds its cap")
        backend_force = torch.as_tensor(
            applied_force_world, dtype=torch.float32, device=view._device
        )
        backend_moment = torch.as_tensor(
            applied_moment_world, dtype=torch.float32, device=view._device
        )

        def submit_once():
            nonlocal submitted_calls
            backend_position = None
            application_point_world = None
            if body_com_application:
                current_position, current_quaternion = part.get_world_pose()
                if hasattr(current_position, "detach"):
                    current_position = current_position.detach().cpu()
                if hasattr(current_quaternion, "detach"):
                    current_quaternion = current_quaternion.detach().cpu()
                current_position = np.asarray(
                    current_position, dtype=np.float64
                )
                current_quaternion = np.asarray(
                    current_quaternion, dtype=np.float64
                )
                application_point_world = (
                    current_position
                    + _quaternion_wxyz_rotation(current_quaternion)
                    @ application_point_object
                )
                backend_position = torch.as_tensor(
                    application_point_world.astype(np.float32).reshape(1, 3),
                    dtype=torch.float32,
                    device=view._device,
                )
            view.apply_forces_and_torques_at_pos(
                forces=backend_force,
                torques=backend_moment,
                positions=backend_position,
                is_global=True,
            )
            submitted_calls += 1
            return application_point_world, current_position if body_com_application else None, current_quaternion if body_com_application else None

        measured_effort = stepper.latest[2][8:] - tare[1:]
        resistive_effort = closing_direction * measured_effort
        regulated_target = np.array(hand_target, copy=True)
        for finger_index, hand_index in enumerate((1, 2, 3)):
            effort_error = (
                required_effort[finger_index]
                - resistive_effort[finger_index]
            )
            joint_delta = closing_direction[finger_index] * float(
                np.clip(
                    effort_error / hand_stiffness,
                    -maximum_finger_increment,
                    maximum_finger_increment,
                )
            )
            regulated_target[hand_index] = float(np.clip(
                hand_target[hand_index] + joint_delta,
                preload_lower[hand_index],
                preload_upper[hand_index],
            ))
        hand_target = regulated_target
        applied_position_world = None
        applied_body_origin_world = None
        applied_body_orientation_world = None

        def submit_and_record_position():
            nonlocal applied_position_world
            nonlocal applied_body_origin_world
            nonlocal applied_body_orientation_world
            (
                applied_position_world,
                applied_body_origin_world,
                applied_body_orientation_world,
            ) = submit_once()

        latest = stepper.advance(
            phase,
            arm_target,
            hand_target,
            pre_step_hook=submit_and_record_position,
        )
        if latest is None or runtime["auditor"].samples[-1]["phase"] != phase:
            return False
        row = runtime["auditor"].samples[-1]
        row["postgrasp_disturbance_input"] = {
            "condition_id": condition["condition_id"],
            "scale": value,
            "force_task_n": applied_force_task.tolist(),
            "moment_task_nm": applied_moment_task.tolist(),
            "force_world_n": applied_force_world.astype(np.float64)[0].tolist(),
            "moment_world_nm": applied_moment_world.astype(np.float64)[0].tolist(),
            "application_point": (
                "CURRENT_TE_BODY_CENTER_OF_MASS"
                if body_com_application
                else "CURRENT_TE_RIGID_BODY_ORIGIN"
            ),
            "application_point_supplier_object_m": (
                None
                if application_point_object is None
                else application_point_object.tolist()
            ),
            "application_point_world_m": (
                None
                if applied_position_world is None
                else applied_position_world.tolist()
            ),
            "application_body_origin_world_m": (
                None
                if applied_body_origin_world is None
                else applied_body_origin_world.tolist()
            ),
            "application_body_orientation_world_wxyz": (
                None
                if applied_body_orientation_world is None
                else applied_body_orientation_world.tolist()
            ),
            "target_rigid_body_path": str(target_prim_path),
            "application_point_world_readback_used": body_com_application,
            "application_point_world_readback_scope": (
                "TEST_LOAD_APPLICATION_ONLY_NOT_GRASP_CONTROL"
                if body_com_application
                else None
            ),
            "is_global": True,
            "submission_tensor_backend": "torch",
            "submission_tensor_device": str(view._device),
            "submission_count_this_physics_step": 1,
        }
        tare_subtracted = latest[2][8:] - tare[1:]
        if float(np.max(np.abs(tare_subtracted))) > effort_abort:
            stepper.abort_reason = (
                "POSTGRASP_DISTURBANCE_HAND_MEASURED_EFFORT_ABORT"
            )
            return False
        return stepper.abort_reason is None

    timing = contract["timing"]
    dt = float(arguments.dynamic_settings["physics_dt_s"])
    ramp_up_steps = round(float(timing["ramp_up_s"]) / dt)
    plateau_steps = round(float(timing.get("plateau_s", 0.0)) / dt)
    ramp_down_steps = round(float(timing["ramp_down_s"]) / dt)
    recovery_steps = round(float(timing["recovery_s"]) / dt)
    execution.update({
        "started": True,
        "condition_id": condition["condition_id"],
        "start_step": int(stepper.step_index),
        "planned_steps": {
            "zero_input_baseline": 1,
            "ramp_up": ramp_up_steps,
            "plateau": plateau_steps,
            "ramp_down": ramp_down_steps,
            "recovery": recovery_steps,
        },
        "wrench_directions_frozen_in_world": True,
        "online_object_pose_readback_used": body_com_application,
        "disturbance_vector_frame_source": (
            contract["coordinate_contract"].get(
                "frozen_world_task_frame_source"
            )
        ),
        "online_object_pose_readback_scope": (
            "TEST_FRAME_AND_COM_APPLICATION_ONLY_NOT_GRASP_CONTROL"
            if body_com_application
            else None
        ),
        "target_rigid_body_path": str(target_prim_path),
        "finger_effort_maintenance_enabled": True,
        "finger_effort_regulation_method": (
            "BIDIRECTIONAL_EFFORT_ERROR_TO_BOUNDED_POSITION_STEP"
        ),
        "required_closing_joint_effort_nm": required_effort.tolist(),
        "finite_preload_limit_goal_rad": preload_limit_goal.tolist(),
        "finite_preload_lower_bound_rad": preload_lower.tolist(),
        "finite_preload_upper_bound_rad": preload_upper.tolist(),
    })
    active = advance_with_wrench("postgrasp_disturbance_baseline", 0.0)
    for index in range(ramp_up_steps):
        if not active:
            break
        scale = control.minimum_jerk_blend((index + 1) / ramp_up_steps)
        active = advance_with_wrench(
            "postgrasp_disturbance_ramp_up", scale
        )
        if active and index + 1 == ramp_up_steps:
            visual_capture = runtime.get("visual_capture")
            if visual_capture is not None:
                visual_capture._capture_once(
                    "05_disturbance_peak", runtime["auditor"].samples[-1]
                )
    for _ in range(plateau_steps):
        if not active:
            break
        active = advance_with_wrench(
            "postgrasp_disturbance_plateau", 1.0
        )
    for index in range(ramp_down_steps):
        if not active:
            break
        scale = 1.0 - control.minimum_jerk_blend(
            (index + 1) / ramp_down_steps
        )
        active = advance_with_wrench(
            "postgrasp_disturbance_ramp_down", scale
        )
    for _ in range(recovery_steps):
        if not active:
            break
        active = advance_with_wrench(
            "postgrasp_disturbance_recovery", 0.0
        )

    execution.update({
        "completed": bool(active and stepper.abort_reason is None),
        "failure_reason": stepper.abort_reason,
        "end_step": int(stepper.step_index - 1),
        "submitted_wrench_call_count": submitted_calls,
        "observed_steps": {
            phase: sum(
                row["phase"] == phase for row in runtime["auditor"].samples
            )
            for phase in POSTGRASP_DISTURBANCE_PHASES
        },
    })
    if execution["completed"] is not True and execution["failure_reason"] is None:
        execution["failure_reason"] = "POSTGRASP_DISTURBANCE_INCOMPLETE"
        grasp_result["failure_reason"] = execution["failure_reason"]
    elif execution["failure_reason"] is not None:
        grasp_result["failure_reason"] = execution["failure_reason"]
    return execution


def _run_controller(runtime, arguments, motion_plan, dynamic):
    robot, active_indices, arm_indices, lower, upper, drive_audit = runtime["robot_data"]
    stepper = control.JointSignalStepper(
        robot=robot, world=runtime["world"], auditor=runtime["auditor"],
        active_indices=active_indices, arm_indices=arm_indices,
        arm_lower_limits=lower, arm_upper_limits=upper,
        settings=dynamic, render=arguments.gui,
        robot_model=runtime["robot_model"],
        payload_model=runtime["payload_model"],
    )
    pregrasp = control.run_pregrasp_sequence(
        stepper,
        motion_plan,
        dynamic,
        initialized_at_pregrasp=arguments.initialize_at_pregrasp,
    )
    grasp = (
        control.run_grasp_lift_sequence(
            stepper, motion_plan, dynamic, pregrasp,
            first_finger_only=arguments.mode == "first-finger-diagnostic")
        if arguments.mode in ("first-finger-diagnostic", "grasp-lift")
        else {"contact_controller": None, "failure_reason": stepper.abort_reason}
    )
    disturbance_execution = _run_postgrasp_disturbance(
        runtime, arguments, stepper, grasp
    )
    outcome = control.controller_outcome(
        stepper, mode=arguments.mode, native_drive_audit=drive_audit,
        pregrasp=pregrasp, grasp=grasp,
    )
    return stepper, outcome, disturbance_execution


def _runtime_record(repository, inputs, runtime):
    scene = runtime["scene"]
    record = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            check=True, capture_output=True,
        ).stdout.strip(),
        "config_path": str(inputs.config.path),
        "config_sha256": file_sha256(inputs.config.path),
        "registered_grasp_sha256": _json_sha256(runtime["registered_grasp"]),
        "control_plan_sha256": _json_sha256(runtime["control_plan"]),
        "runtime_resources_sha256": file_sha256(runtime["runtime_resources_path"]),
        "capacity_audit_sha256": runtime["capacity_audit_sha256"],
        "scene_evidence_paths": [str(path) for path in scene["evidence_paths"]],
        "scene_evidence_sha256": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in scene["evidence_paths"]
        },
        "robot_asset_sha256": file_sha256(runtime["robot_asset"]),
        "object_asset_sha256": file_sha256(scene["object_asset"]),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "controller.py", "run_grasp_lift.py", "evaluate_run.py",
                "engine_health.py",
            )
        },
    }
    disturbance = runtime.get("postgrasp_disturbance")
    if disturbance is not None:
        record["postgrasp_disturbance_evidence"] = {
            "panel_path": disturbance["panel_path"],
            "panel_sha256": disturbance["panel_sha256"],
            "condition_id": disturbance["condition"]["condition_id"],
            "nominal_qualification_evaluation_path": disturbance[
                "nominal_qualification_evaluation_path"
            ],
            "nominal_qualification_evaluation_sha256": disturbance[
                "nominal_qualification_evaluation_sha256"
            ],
        }
    return record


def _split_plug_relative_motion_summary(runtime) -> dict[str, object]:
    """Measure the joint's actual relative motion after simulation only."""

    scene = runtime["scene"]
    samples = runtime["auditor"].samples
    if scene.get("relative_motion_model") != (
        "BODY_PLUS_UNLIMITED_COAXIAL_COUPLING_NUT_REVOLUTE"
    ):
        return {"status": "NOT_APPLICABLE_FUSED_OBJECT", "online_control_used": False}
    if len(scene["part_prim_paths"]) != 2 or not samples:
        return {"status": "UNAVAILABLE", "online_control_used": False}

    def pose_matrix(row, index):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quaternion_wxyz_rotation(
            row["object_part_orientations_wxyz"][index]
        )
        matrix[:3, 3] = np.asarray(
            row["object_part_positions_m"][index], dtype=np.float64
        )
        return matrix

    relative = [
        np.linalg.inv(pose_matrix(row, 0)) @ pose_matrix(row, 1)
        for row in samples
    ]
    reference = relative[0]
    translations = np.asarray(
        [matrix[:3, 3] - reference[:3, 3] for matrix in relative],
        dtype=np.float64,
    )
    delta_rotations = [reference[:3, :3].T @ matrix[:3, :3] for matrix in relative]
    axis_tilts = np.asarray(
        [
            math.acos(
                float(np.clip(rotation[2, 2], -1.0, 1.0))
            )
            for rotation in delta_rotations
        ],
        dtype=np.float64,
    )
    wrapped_twists = np.asarray(
        [math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
         for rotation in delta_rotations],
        dtype=np.float64,
    )
    unwrapped_twists = np.unwrap(wrapped_twists)
    translation_norms = np.linalg.norm(translations, axis=1)
    resistance = dict(scene.get("joint_rotational_resistance") or {})
    resistance_authored = bool(
        float(resistance.get("assumed_resisting_torque_nm", 0.0)) > 0.0
    )
    return {
        "status": "COMPLETE",
        "part_order": list(scene["part_prim_paths"]),
        "reference_rule": "FIRST_POST_STEP_SAMPLE",
        "sample_count": len(samples),
        "maximum_relative_translation_change_m": float(
            np.max(translation_norms)
        ),
        "final_relative_translation_change_m": float(translation_norms[-1]),
        "maximum_relative_translation_components_m": np.max(
            np.abs(translations), axis=0
        ).tolist(),
        "maximum_noncoaxial_axis_tilt_rad": float(np.max(axis_tilts)),
        "final_noncoaxial_axis_tilt_rad": float(axis_tilts[-1]),
        "final_unwrapped_coaxial_rotation_rad": float(unwrapped_twists[-1]),
        "minimum_unwrapped_coaxial_rotation_rad": float(
            np.min(unwrapped_twists)
        ),
        "maximum_unwrapped_coaxial_rotation_rad": float(
            np.max(unwrapped_twists)
        ),
        "total_unwrapped_coaxial_rotation_range_rad": float(
            np.ptp(unwrapped_twists)
        ),
        "joint_drive_or_friction_authored": resistance_authored,
        "rotational_resistance": resistance,
        "online_control_used": False,
        "evidence_limit": (
            "MEASURES_RESPONSE_TO_AN_ASSUMED_SYMMETRIC_RESISTING_TORQUE;"
            "DOES_NOT_ESTABLISH_THIS_SPECIMENS_TRUE_UNLOADED_TORQUE"
            if resistance_authored
            else "MEASURES_IDEAL_REVOLUTE_CONSTRAINT_AND_FREE_RELATIVE_ROTATION;"
            "DOES_NOT_MODEL_UNMEASURED_SPECIMEN_ROTATIONAL_RESISTANCE"
        ),
    }


def _split_plug_contact_policy_summary(runtime, evaluation) -> dict[str, object]:
    scene = runtime["scene"]
    if scene.get("relative_motion_model") != (
        "BODY_PLUS_UNLIMITED_COAXIAL_COUPLING_NUT_REVOLUTE"
    ):
        return {"status": "NOT_APPLICABLE_FUSED_OBJECT"}
    legal = set(map(str, scene.get("legal_grasp_contact_paths", ())))
    slip = evaluation.get("fingertip_contact_surface_slip", {})
    fingers = []
    for row in slip.get("per_finger", ()):
        contacted = set(map(str, row.get("contacted_object_part_paths", ())))
        fingers.append(
            {
                "terminal_link": row.get("terminal_link"),
                "contacted_part_paths": sorted(contacted),
                "contacted_coupling_nut": bool(contacted & legal),
                "contacted_body_or_other_part": bool(contacted - legal),
                "nut_only": bool(contacted and contacted <= legal),
            }
        )
    observed = {path for row in fingers for path in row["contacted_part_paths"]}
    all_three_nut_only = bool(
        len(fingers) == 3 and all(row["nut_only"] for row in fingers)
    )
    return {
        "status": "COMPLETE" if observed else "NO_TERMINAL_CONTACT",
        "required_contact_part_paths": sorted(legal),
        "observed_contact_part_paths": sorted(observed),
        "per_finger": fingers,
        "all_three_fingers_contacted_coupling_nut_only": all_three_nut_only,
        "body_contact_observed": bool(observed - legal),
        "online_control_used": False,
    }


def _apply_split_plug_contact_policy(
    evaluation: dict[str, object], policy: Mapping[str, object]
) -> None:
    """Make nut-only contact part of a split-plug disturbance decision."""

    disturbance = evaluation.get("postgrasp_disturbance")
    if not isinstance(disturbance, dict) or disturbance.get("requested") is not True:
        return
    nut_only = bool(
        policy.get("all_three_fingers_contacted_coupling_nut_only") is True
        and policy.get("body_contact_observed") is False
    )
    disturbance["coupling_nut_only_contact_pass"] = nut_only
    if not nut_only:
        disturbance["core_condition_pass"] = False
        disturbance["status"] = "FAIL"
        disturbance.setdefault("failure_reasons", []).append(
            "FINGERTIP_CONTACTED_BODY_OR_NON_NUT_PART"
        )


def _finish_run(repository, inputs, runtime, trace, outcome):
    trace["controller_outcome"] = outcome
    trace["samples"] = runtime["auditor"].samples
    split_relative_motion = _split_plug_relative_motion_summary(runtime)
    trace["split_plug_relative_motion"] = split_relative_motion
    visual_consumption = trace.get("visual_transport_target_consumption")
    if isinstance(visual_consumption, dict) and visual_consumption.get("provided"):
        visual_consumption["controller_execution_observed"] = bool(
            runtime["auditor"].samples
        )
    trace["audit_roots"] = {"robot": ROBOT_ROOT, **runtime["scene"]["roots"]}
    visual_capture = runtime.get("visual_capture")
    if visual_capture is not None:
        trace["visual_evidence"] = {
            "schema_version": "carts_grasp_v2_visual_evidence_v1",
            "post_step_observation_only": True,
            "returned_to_controller": False,
            "records": visual_capture.records,
        }
    trace["runtime"] = _runtime_record(repository, inputs, runtime)
    trace["identity_hash_check_pass"] = identity_hashes_match(trace)
    engine_runtime = runtime["engine_monitor"].summary()
    engine_runtime["gpu_backend_pass"] = trace["physics_backend"]["pass"]
    evaluation = evaluate_trace(
        trace, robot_asset_path=runtime["robot_asset"], inputs=inputs
    )
    evaluation["split_plug_relative_motion"] = split_relative_motion
    evaluation["split_plug_grasp_contact_policy"] = (
        _split_plug_contact_policy_summary(runtime, evaluation)
    )
    _apply_split_plug_contact_policy(
        evaluation, evaluation["split_plug_grasp_contact_policy"]
    )
    return trace, evaluation, engine_runtime


def _execute(
    repository, arguments, output, inputs, grasp, scene_entry, motion_plan,
    trace, simulation_app,
):
    if arguments.mode == "isolated-hand":
        return _execute_isolated(
            repository, arguments, output, inputs, scene_entry, motion_plan, trace
        )
    runtime = _create_runtime(
        repository, arguments, inputs, grasp, scene_entry, motion_plan, trace,
        simulation_app, output,
    )
    if arguments.mode == VISION_GRASP_SERVO_MODE:
        return runtime["visual_grasp_result"]
    if arguments.mode in (SAME_RESET_RGBD_MODE, VISION_HIGH_REOBSERVE_MODE):
        return int(runtime["same_reset_rgbd_exit_code"])
    dynamic = arguments.dynamic_settings
    if arguments.capture_visual_evidence:
        runtime["visual_capture"] = _VisualEvidenceCapture(
            world=runtime["world"], auditor=runtime["auditor"], output=output,
            physics_dt_s=float(dynamic["physics_dt_s"]),
        )
    _, outcome, disturbance_execution = _run_controller(
        runtime, arguments, motion_plan, dynamic
    )
    trace["postgrasp_disturbance_execution"] = disturbance_execution
    if arguments.capture_visual_evidence and arguments.mode == "grasp-lift":
        runtime["visual_capture"].capture_run_end()
    return _finish_run(repository, inputs, runtime, trace, outcome)


def _write_failure(output: Path, error: Exception) -> None:
    payload = {"error_type": type(error).__name__, "error": str(error),
               "traceback": traceback.format_exc(), "hardware_authorized": False}
    (output / "failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    repository = Path(__file__).resolve().parents[4]
    arguments = _arguments(repository)
    output = Path(arguments.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, grasp, scene_entry, motion_plan = _load_plan_inputs(
        repository, arguments)
    dynamic = arguments.dynamic_settings
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": not (arguments.gui or arguments.capture_visual_evidence),
        "multi_gpu": False,
        "active_gpu": 0, "physics_gpu": 0, "fast_shutdown": True,
    })
    engine_log_path = current_engine_log_path()
    trace = (
        _initial_isolated_trace(arguments, inputs, grasp, motion_plan, dynamic)
        if arguments.mode == "isolated-hand"
        else _initial_same_reset_rgbd_trace(arguments, inputs, grasp, dynamic)
        if arguments.mode in (SAME_RESET_RGBD_MODE, *VISION_MOTION_MODES)
        else _initial_trace(arguments, inputs, grasp, motion_plan, dynamic)
    )
    try:
        result = _execute(repository, arguments, output, inputs, grasp,
                          scene_entry, motion_plan, trace, simulation_app)
    except Exception as error:
        _write_failure(output, error)
        traceback.print_exc()
        simulation_app.close(exit_code=1)
        return 1
    if isinstance(result, int):
        simulation_app.close(exit_code=result)
        return result
    try:
        trace, evaluation, engine_runtime = result
        engine_runtime["engine_log_sync"] = synchronize_engine_log(engine_log_path)
        evaluation = finalize_engine_evaluation(evaluation, engine_runtime, engine_log_path)
        if not arguments.omit_trace_json:
            (output / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if "visual_evidence" in trace:
            (output / "visual_evidence.json").write_text(
                json.dumps(trace["visual_evidence"], ensure_ascii=False, indent=2)
                + "\n", encoding="utf-8")
        (output / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        if arguments.postgrasp_disturbance is not None:
            exit_pass = bool(
                evaluation["nominal_research_dynamic_pass"]
                and evaluation["postgrasp_disturbance"].get(
                    "core_condition_pass"
                ) is True
            )
        else:
            key = (
                "visual_servo_grasp_physical_pass"
                if arguments.mode == VISION_GRASP_SERVO_MODE
                else
                "accepted_preflight_pass"
                if arguments.mode == "preflight"
                else "first_finger_diagnostic_pass"
                if arguments.mode == "first-finger-diagnostic"
                else "nominal_research_dynamic_pass"
            )
            exit_pass = bool(
                evaluation[key]
                and (
                    arguments.mode != VISION_GRASP_SERVO_MODE
                    or (
                        evaluation.get("engine_health_pass") is True
                        and evaluation.get("identity_hash_check_pass") is True
                    )
                )
            )
        exit_code = 0 if exit_pass else 2
    except Exception as error:
        _write_failure(output, error)
        traceback.print_exc()
        exit_code = 1
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
