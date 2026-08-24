"""Write one compact offline report set and a diagnostic candidate preview."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy

from kcg_connector.grasp.carts_v2.pipeline import OfflinePipelineResult
from kcg_connector.grasp.robust.object_model import file_sha256
from kcg_connector.grasp.carts_v2.task_quality import (
    minimum_jerk_peak_acceleration,
)


def _array_sha256(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _plain_metric(value: object) -> str:
    return "不可用" if value is None else f"{float(value):.3f}"


def _quality_map(result: OfflinePipelineResult):
    return {row.candidate_id: row for row in result.task_quality_results}


def _filter_map(result: OfflinePipelineResult):
    return {row.candidate_id: row for row in result.raw_fast_filter_results}


def _candidate_rows(result: OfflinePipelineResult):
    quality = _quality_map(result)
    filters = _filter_map(result)
    formal_ids = {row.candidate_id for row in result.candidates}
    for prediction in result.raw_closure_predictions:
        candidate_id = prediction.seed.candidate_id
        metric = quality.get(candidate_id)
        fast_filter = filters[candidate_id]
        yield {
            "candidate_id": candidate_id,
            "source_sample_index": prediction.seed.source_sample_index,
            "anchor_face_index": prediction.seed.anchor_face_index,
            "anchor_position_object_m": " ".join(
                f"{value:.9g}" for value in prediction.seed.anchor_position_object_m
            ),
            "closure_status": prediction.status,
            "contact_face_indices": " ".join(
                str(contact.object_face_index) for contact in prediction.contacts
            ),
            "contact_positions_object_m": " | ".join(
                " ".join(f"{value:.9g}" for value in contact.object_position_m)
                for contact in prediction.contacts
            ),
            "fast_filter": fast_filter.status,
            "fast_reasons": " | ".join(fast_filter.reasons),
            "formal_candidate_selected": candidate_id in formal_ids,
            "formal_selection_status": (
                "SELECTED_AFTER_SWEEP_DIVERSITY"
                if candidate_id in formal_ids
                else (
                    "REJECTED_BEFORE_DIVERSITY"
                    if fast_filter.status == "FAST_REJECT"
                    else result.diversity_rejection_reasons.get(
                        candidate_id, "PATH_SAFE_SELECTION_REASON_MISSING"
                    )
                )
            ),
            "sequential_closure_sweep_pass": (
                fast_filter.sequential_closure_sweep_pass
            ),
            "minimum_table_clearance_m": fast_filter.minimum_table_clearance_m,
            "minimum_clearance_link": fast_filter.minimum_clearance_link,
            "minimum_clearance_finger_stage": (
                fast_filter.minimum_clearance_finger_stage
            ),
            "minimum_clearance_joint_position_rad": " ".join(
                f"{value:.9g}"
                for value in fast_filter.minimum_clearance_joint_position_rad
            ),
            "checked_state_count": fast_filter.checked_state_count,
            "maximum_joint_increment_rad": fast_filter.maximum_joint_increment_rad,
            "endpoint_only_table_clearance_m": (
                fast_filter.endpoint_only_table_clearance_m
            ),
            "first_table_violation_clearance_m": (
                fast_filter.first_table_violation_clearance_m
            ),
            "first_table_violation_link": fast_filter.first_table_violation_link,
            "first_table_violation_finger_stage": (
                fast_filter.first_table_violation_finger_stage
            ),
            "unresolved_checks": " | ".join(fast_filter.unresolved_checks),
            "task_status": "NOT_EVALUATED" if metric is None else metric.status,
            "worst_task_margin": "" if metric is None else metric.worst_task_margin,
            "lower_tail_mean_margin": (
                "" if metric is None else metric.lower_tail_mean_margin
            ),
            "required_peak_normal_force_n": (
                "" if metric is None else metric.required_peak_normal_force_n
            ),
            "maximum_joint_load_utilization": (
                "" if metric is None else metric.maximum_joint_load_utilization
            ),
            "maximum_generalized_joint_torque_nm": (
                ""
                if metric is None
                else metric.maximum_generalized_joint_torque_nm
            ),
            "wrist_load_utilization": (
                "" if metric is None else metric.wrist_load_utilization
            ),
            "sensitivity": "" if metric is None else metric.sensitivity,
            "task_failure_reason": "" if metric is None else metric.failure_reason,
            "nominal_balance_infeasible_count": (
                "" if metric is None else metric.nominal_balance_infeasible_count
            ),
            "evidence": "OFFLINE_RESEARCH_NOT_FORMAL",
        }


def _selected_json(result: OfflinePipelineResult, selected_rows) -> list[dict[str, object]]:
    rows = []
    exact_by_id = {
        row.candidate_id: row for row in result.exact_validation_results
    }
    for selected in selected_rows:
        quality = selected.task_quality
        prediction = selected.prediction
        exact = exact_by_id.get(prediction.seed.candidate_id)
        if selected.selection_status == "EXECUTABLE_CANDIDATE" and exact is None:
            raise ValueError("executable candidate lacks exact-validation result")
        rows.append(
            {
                "rank": selected.rank,
                "candidate_id": prediction.seed.candidate_id,
                "selection_status": selected.selection_status,
                "selection_scope": "OFFLINE_ELIGIBLE_FOR_BOUND_PREFLIGHT_NOT_DYNAMIC_PASS",
                "task_status": quality.status,
                "three_effective_pad_contacts": len(prediction.contacts) == 3,
                "worst_task_margin": quality.worst_task_margin,
                "lower_tail_mean_margin": quality.lower_tail_mean_margin,
                "required_peak_normal_force_n": quality.required_peak_normal_force_n,
                "required_peak_normal_force_interpretation": (
                    "UNAVAILABLE_TASK_SCALE_INFEASIBLE_OR_UNRESOLVED"
                    if quality.required_peak_normal_force_n is None
                    else "FINITE_AT_LAMBDA_ONE"
                ),
                "maximum_joint_load_utilization": (
                    quality.maximum_joint_load_utilization
                ),
                "maximum_joint_load_utilization_source": (
                    "UNKNOWN_URDF_EFFORT_UNCALIBRATED"
                ),
                "maximum_generalized_joint_torque_nm": (
                    quality.maximum_generalized_joint_torque_nm
                ),
                "wrist_load_utilization": quality.wrist_load_utilization,
                "wrist_load_utilization_source": "UNKNOWN",
                "path_minimum_clearance_m": selected.path_minimum_clearance_m,
                "path_minimum_clearance_interpretation": (
                    "SAMPLED_HAND_LINK_TO_FINITE_TABLE_NOT_CONTINUOUS_OR_ARM_COMPLETE"
                ),
                "minimum_clearance_link": (
                    selected.fast_filter.minimum_clearance_link
                ),
                "minimum_clearance_stage": (
                    selected.fast_filter.minimum_clearance_finger_stage
                ),
                "exact_validation_status": (
                    "NOT_RUN_DIAGNOSTIC_ONLY" if exact is None else exact.status
                ),
                "exact_validation_reason": "" if exact is None else exact.reason,
                "exact_backend_invoked": False if exact is None else exact.backend_invoked,
                "sensitivity": quality.sensitivity,
                "failure_reason": quality.failure_reason,
                "nominal_balance_infeasible_count": (
                    quality.nominal_balance_infeasible_count
                ),
                "fast_filter": selected.fast_filter.status,
                "sequential_closure_sweep_pass": (
                    selected.fast_filter.sequential_closure_sweep_pass
                ),
                "unresolved_checks": list(selected.fast_filter.unresolved_checks),
                "offline_task_gate_passed": selected.offline_task_gate_passed,
                "research_dynamic_gate_passed": False,
                "formal_dynamic_eligible": False,
                "control_plan": {
                    "transform_semantics": "OBJECT_FROM_HAND_BASE",
                    "object_from_hand_row_major": list(
                        prediction.seed.object_from_hand
                    ),
                    "pregrasp_joint_positions_rad": list(
                        prediction.seed.pregrasp_joint_positions_rad
                    ),
                    "final_joint_positions_rad": list(
                        prediction.final_joint_positions_rad
                    ),
                    "final_closure_phases": list(
                        prediction.final_closure_phases
                    ),
                    "closing_order": list(
                        result.inputs.config.section("closure_prediction")[
                            "closing_order"
                        ]
                    ),
                    "predicted_contacts_object": [
                        {
                            "pad_name": contact.pad_name,
                            "position_m": list(contact.object_position_m),
                            "face_index": contact.object_face_index,
                        }
                        for contact in prediction.contacts
                    ],
                },
            }
        )
    return rows


def _write_preview(result: OfflinePipelineResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    mesh = result.inputs.object_contract.model.mesh
    centroids = mesh.face_centroids_m
    allowed = centroids[result.inputs.face_roles.face_is_allowed]
    background_stride = max(1, len(centroids) // 5000)
    allowed_stride = max(1, len(allowed) // 3000)
    anchors = np.asarray(
        [candidate.anchor_position_object_m for candidate in result.candidates]
    )
    figure = plt.figure(figsize=(9, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    background = centroids[::background_stride]
    axis.scatter(*background.T, s=0.3, c="#777777", alpha=0.12)
    axis.scatter(*allowed[::allowed_stride].T, s=1.0, c="#1976d2", alpha=0.40)
    if len(anchors):
        axis.scatter(
            *anchors.T,
            s=20,
            c="#ff9800",
            label=f"{len(result.candidates)} candidate anchors",
        )
    colors = ("#d32f2f", "#7b1fa2", "#388e3c")
    preview_rows = result.executable_candidates or result.diagnostic_candidates
    preview_label = "Executable" if result.executable_candidates else "Diagnostic-only"
    for color, selected in zip(colors, preview_rows):
        contacts = np.asarray(
            [row.object_position_m for row in selected.prediction.contacts]
        )
        axis.scatter(
            *contacts.T,
            s=70,
            c=color,
            marker="x",
            label=f"{preview_label} Top-{selected.rank} contacts",
        )
    axis.set_xlabel("object x / m")
    axis.set_ylabel("object y / m")
    axis.set_zlabel("object z / m")
    axis.set_title(f"CARTS-Grasp V2 offline preview\n{result.inputs.object_contract.object_id}")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _result_document(result: OfflinePipelineResult) -> dict[str, object]:
    filters = result.raw_fast_filter_results
    task_settings = result.inputs.config.section("task_quality")
    dynamic_settings = result.inputs.config.section("dynamic")
    mass = float(result.inputs.object_contract.model.mass_kg)
    mass_error = float(task_settings["mass_relative_error"])
    gravity = float(task_settings["gravity_acceleration_m_s2"])
    lift_acceleration = minimum_jerk_peak_acceleration(
        float(dynamic_settings["lift_distance_m"]),
        float(dynamic_settings["lift_duration_s"]),
    )
    force_scale = mass * gravity
    moment_scale = force_scale * float(
        result.inputs.object_contract.characteristic_radius_m
    )
    return {
        "schema_version": "carts_grasp_v2_offline_result_v2",
        "object_id": result.inputs.object_contract.object_id,
        "config_path": str(result.inputs.config.path),
        "config_sha256": file_sha256(result.inputs.config.path),
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "face_role_method": result.inputs.face_roles.method,
        "allowed_face_count": int(np.sum(result.inputs.face_roles.face_is_allowed)),
        "raw_surface_seed_count": len(result.raw_candidates),
        "candidate_count": len(result.candidates),
        "candidate_selection_method": (
            "FIXED_384_SURFACE_POOL_THEN_CLOSURE_SWEEP_THEN_DIVERSITY_MAX_48"
        ),
        "closure_survive_count": sum(
            row.status == "CLOSURE_SURVIVE"
            for row in result.raw_closure_predictions
        ),
        "sequential_closure_sweep_survive_count": sum(
            row.sequential_closure_sweep_pass for row in filters
        ),
        "table_sweep_reject_count": sum(
            "INTERMEDIATE_SEQUENTIAL_CLOSURE_HAND_TABLE_SWEEP" in row.reasons
            for row in filters
        ),
        "pregrasp_or_endpoint_table_reject_count": sum(
            "HAND_TABLE_PENETRATION" in row.reasons for row in filters
        ),
        "fast_survive_count": sum(row.status == "FAST_SURVIVE" for row in filters),
        "fast_filter_table_clearance": {
            "state_rule": result.inputs.config.section("fast_filter")[
                "hand_table_state_rule"
            ],
            "penetration_tolerance_m": result.inputs.config.section("fast_filter")[
                "table_penetration_tolerance_m"
            ],
            "tolerance_source": result.inputs.config.section("fast_filter")[
                "table_penetration_tolerance_source"
            ],
            "claim_limit": "SAMPLED_FAST_REJECT_NOT_CONTINUOUS_COLLISION_PROOF",
        },
        "task_survive_count": sum(
            row.status == "TASK_SURVIVE" for row in result.task_quality_results
        ),
        "executable_top_k_count": len(result.executable_candidates),
        "diagnostic_candidate_count": sum(
            row.status != "TASK_SURVIVE" for row in result.task_quality_results
        ),
        "selected_executable_candidate": (
            None
            if not result.executable_candidates
            else result.executable_candidates[0].prediction.seed.candidate_id
        ),
        "offline_selection_status": (
            "EXECUTABLE_CANDIDATE_AVAILABLE"
            if result.executable_candidates
            else "NO_FEASIBLE_CANDIDATE_ANALYSIS"
        ),
        "exact_validation": {
            "top_k_count": len(result.exact_validation_results),
            "backend_invocation_count": sum(
                row.backend_invoked for row in result.exact_validation_results
            ),
            "all_formally_resolved": bool(result.exact_validation_results)
            and all(
                row.status == "CERTIFIED_FREE"
                for row in result.exact_validation_results
            ),
        },
        "scenario_design": {
            "method": "SCRAMBLED_SOBOL",
            "shape": list(result.scenario_design.shape),
            "sha256_float64_c_order": _array_sha256(result.scenario_design),
            "values": result.scenario_design.tolist(),
        },
        "lambda_one_task_load": {
            "semantics": (
                "NOMINAL_GRAVITY_PLUS_PEAK_LIFT_ACCELERATION_AND_ONE_"
                "PREREGISTERED_FORCE_OR_MOMENT_DISTURBANCE_VERTEX"
            ),
            "registered_mass_kg": mass,
            "nominal_force_magnitude_n": mass * (gravity + lift_acceleration),
            "disturbance_force_scale_n": force_scale,
            "disturbance_moment_scale_nm": moment_scale,
            "study_mass_relative_range": [1.0 - mass_error, 1.0 + mass_error],
            "lift_peak_acceleration_m_s2": lift_acceleration,
        },
        "runtime": {
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "timings_s": result.timings_s,
        "executable_candidates": _selected_json(
            result, result.executable_candidates
        ),
        "diagnostic_candidates": _selected_json(
            result, result.diagnostic_candidates
        ),
    }


def _write_summary(
    result: OfflinePipelineResult,
    document: dict[str, object],
    summary_path: Path,
) -> None:
    executable = document["executable_candidates"]
    diagnostic = document["diagnostic_candidates"]
    if executable:
        best = executable[0]
        selection_text = (
            f"可执行第一名：{best['candidate_id']}，最差载荷余量 "
            f"{_plain_metric(best['worst_task_margin'])}，较差场景平均余量 "
            f"{_plain_metric(best['lower_tail_mean_margin'])}。\n\n"
        )
    else:
        diagnostic_id = "无" if not diagnostic else diagnostic[0]["candidate_id"]
        selection_text = (
            "当前没有同时通过闭合路径与任务载荷门的可执行候选；"
            f"诊断列表第一项为 {diagnostic_id}，不得执行完整抬升。\n\n"
        )
    summary_path.write_text(
        "# CARTS-Grasp V2 离线摘要\n\n"
        f"一句话结论：384 个固定原始种子中，完整顺序闭合扫掠保留 "
        f"{document['sequential_closure_sweep_survive_count']} 个，"
        f"任务载荷门通过 {document['task_survive_count']} 个。\n\n"
        "机器人/连接器实际发生：本阶段只运行离线几何与力学，尚未执行抓取。\n\n"
        + selection_text
        + "证据等级：离线算法；不是动态仿真或正式动态验收。\n\n"
        "仍不确定：机械臂逆解、非指腹全路径碰撞和腕部负载尚未闭合。\n",
        encoding="utf-8",
    )


def write_offline_report(
    result: OfflinePipelineResult, output_directory: Path | str
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "candidates.csv"
    rows = list(_candidate_rows(result))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    document = _result_document(result)
    json_path = output / "result.json"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = output / "SUMMARY_CN.md"
    _write_summary(result, document, summary_path)
    preview_path = output / "candidate_preview.png"
    _write_preview(result, preview_path)
    return {
        "candidates_csv": csv_path,
        "result_json": json_path,
        "summary_cn": summary_path,
        "preview_png": preview_path,
    }


__all__ = ["write_offline_report"]
