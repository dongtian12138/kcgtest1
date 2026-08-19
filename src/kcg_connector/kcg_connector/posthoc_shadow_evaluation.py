"""Posthoc-only evaluation for the H0 T_HP shadow archive.

Reads the frozen run report and shadow result without feeding anything back to
the formal estimator.  It records translation/rotation error, C2 cost tie,
single-episode covariance overconfidence and pose-valid flags only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import proxy_cad_points
from kcg_connector.d38999_inhand_multiview import (
    c2_action_pose6,
    matrix_pose,
    pose_matrix,
)
from kcg_connector.postgrasp_shadow_estimator import (
    _ResidualProblem,
    estimate_postgrasp_T_HP,
    load_formal_archive,
)

SCHEMA_VERSION = "kcg_d38999_posthoc_shadow_evaluation_v1"


def _c2_equivalent_matrices(truth_4x4: np.ndarray) -> list[np.ndarray]:
    rz_pi = Rotation.from_euler("z", math.pi).as_matrix()
    transformed = np.asarray(truth_4x4, dtype=np.float64).copy()
    transformed[:3, :3] = truth_4x4[:3, :3] @ rz_pi
    return [np.asarray(truth_4x4, dtype=np.float64), transformed]


def _nearest_c2_error(
    hypothesis_4x4: np.ndarray, truth_4x4: np.ndarray
) -> dict[str, Any]:
    translation = float(
        np.linalg.norm(
            hypothesis_4x4[:3, 3] - np.asarray(truth_4x4)[:3, 3]
        )
    )
    rotation_candidates = []
    for truth_equiv in _c2_equivalent_matrices(truth_4x4):
        relative = hypothesis_4x4[:3, :3].T @ truth_equiv[:3, :3]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        rotation_candidates.append(
            (float(np.linalg.norm(rotvec)), rotvec.tolist())
        )
    rotation, rotvec = min(rotation_candidates, key=lambda item: item[0])
    return {
        "translation_error_m": translation,
        "rotation_geodesic_error_rad": rotation,
        "rotation_geodesic_error_deg": math.degrees(rotation),
        "relative_rotvec_rad": rotvec,
    }


def evaluate_posthoc_shadow(
    *,
    report_path: Path | str,
    shadow_result_path: Path | str,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    shadow = json.loads(Path(shadow_result_path).read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("grasp report did not pass")
    truth = np.asarray(report["posthoc_t_hand_plug_actual"], dtype=np.float64)
    hypotheses = shadow.get("c2", {}).get("hypotheses", [])
    if not hypotheses:
        return {
            "schema_version": SCHEMA_VERSION,
            "truth_scope": "POSTHOC_TRUTH_ONLY",
            "decision": "INSUFFICIENT_SHADOW_HYPOTHESES",
        }
    evaluations = []
    for hypothesis in hypotheses:
        pose = pose_matrix(
            np.asarray(hypothesis["T_hand_plug_xyz_rpy"], dtype=np.float64)
        )
        error = _nearest_c2_error(pose, truth)
        rotvec = np.asarray(error["relative_rotvec_rad"], dtype=np.float64)
        axis_tilt = float(np.linalg.norm(rotvec[:2]))
        rz = float(rotvec[2])
        local_rz_mod_pi = abs((rz + math.pi / 2.0) % math.pi - math.pi / 2.0)
        c2_modulo_rotation = float(
            np.linalg.norm([rotvec[0], rotvec[1], local_rz_mod_pi])
        )
        error.update(
            {
                "axis_tilt_rad": axis_tilt,
                "local_rz_mod_pi_rad": local_rz_mod_pi,
                "c2_modulo_rotation_rad": c2_modulo_rotation,
            }
        )
        covariance = np.asarray(hypothesis.get("covariance_6x6"), dtype=np.float64)
        std = np.sqrt(np.diag(covariance)[3:6])
        overconfidence = bool(
            any(
                float(sigma) > 0.0
                and error["rotation_geodesic_error_rad"] > 3.0 * float(sigma)
                for sigma in std
            )
            or error["translation_error_m"]
            > 3.0 * math.sqrt(float(np.trace(covariance[:3, :3])))
        )
        evaluations.append(
            {
                "id": hypothesis["id"],
                "posthoc_error": error,
                "cost": hypothesis.get("cost"),
                "residual_rms": hypothesis.get("residual_rms"),
                "condition_number": hypothesis.get("condition_number"),
                "angle_1sigma_rad": std.tolist(),
                "angle_1sigma_deg": np.degrees(std).tolist(),
                "overconfidence_single_episode": overconfidence,
            }
        )
    costs = [float(item["cost"]) for item in evaluations if item["cost"] is not None]
    cost_delta = (
        None if len(costs) < 2 else abs(costs[0] - costs[1])
    )
    cost_scale = (
        max(costs) if costs and max(costs) > 0.0 else 1.0
    )
    c2_cost_tie = bool(
        cost_delta is not None and cost_delta <= 1.0e-6 * cost_scale
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "formal_pose_valid_claim": shadow.get("pose_valid"),
        "optimizer_converged_claim": shadow.get("optimizer_converged"),
        "c2_resolution_claim": shadow.get("c2", {}).get("resolution"),
        "c2_cost_tie_posthoc": c2_cost_tie,
        "c2_cost_delta": cost_delta,
        "hypotheses": evaluations,
        "overconfidence_single_episode": bool(
            any(item["overconfidence_single_episode"] for item in evaluations)
        ),
        "single_episode_coverage_calibration": "NOT_ATTEMPTED_SINGLE_SAMPLE",
        "truth_feedback_to_formal_estimator": False,
        "control_authorized": False,
    }


def replay_shadow_estimate(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    plug_feature_set: str = "mating_only",
) -> dict[str, Any]:
    """Replay the current estimator on an existing formal archive."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate(
        (matrix_pose(nominal_hp), np.zeros(6, dtype=np.float64))
    )
    return estimate_postgrasp_T_HP(
        views, initial, plug_feature_set=plug_feature_set
    )


def run_T_HP_view_ab(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Per-view-group offline A/B: wrist only, fixed only, wrist+fixed."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate(
        (matrix_pose(nominal_hp), np.zeros(6, dtype=np.float64))
    )
    wrist = [view for view in views if view.group == "postgrasp_inhand_views"]
    fixed = [view for view in views if view.group == "fixed_world_camera_views"]
    groups = {"A_wrist_v0_only": wrist, "B_fixed_world_only": fixed, "C_wrist_plus_fixed": views}
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path_obj = Path(report_path)
    summary = {}
    for name, group_views in groups.items():
        if not group_views:
            summary[name] = {"status": "EMPTY_GROUP"}
            continue
        replay = estimate_postgrasp_T_HP(group_views, initial)
        replay_path = output_root / f"replay_{name}.json"
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        evaluation = evaluate_posthoc_shadow(
            report_path=report_path_obj,
            shadow_result_path=replay_path,
        )
        best = evaluation["hypotheses"][0]
        summary[name] = {
            "status": replay["status"],
            "success": replay["success"],
            "optimizer_converged": replay["optimizer_converged"],
            "pose_valid": replay["pose_valid"],
            "reject_reason": replay.get("reject_reason"),
            "residual_rms": best.get("residual_rms"),
            "condition_number": best.get("condition_number"),
            "translation_error_mm": best["posthoc_error"]["translation_error_m"]
            * 1000.0,
            "rotation_error_deg": best["posthoc_error"][
                "rotation_geodesic_error_deg"
            ],
            "overconfidence_single_episode": best[
                "overconfidence_single_episode"
            ],
            "replay_path": str(replay_path),
        }
    ab = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "groups": summary,
        "decision": "NO_GROUP_POSE_VALID",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    (output_root / "view_ab_summary.json").write_text(
        json.dumps(ab, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return ab


def run_occlusion_policy_ab(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """A0 baseline / A1 ignore foreground / A2 ignore foreground+CAD occluder."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate(
        (matrix_pose(nominal_hp), np.zeros(6, dtype=np.float64))
    )
    variants = [
        ("A0_baseline", "baseline", "global"),
        ("A1_ignore_foreground_occluded", "ignore_foreground_occluded", "global"),
        ("A2_ignore_foreground_and_cad_occluder", "ignore_foreground_and_cad_occluder", "global"),
        ("B1_depth_gated_edge", "baseline", "depth_gated"),
    ]
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path_obj = Path(report_path)
    summary = {}
    for name, policy, edge_policy in variants:
        replay = estimate_postgrasp_T_HP(
            views,
            initial,
            occlusion_policy=policy,
            edge_policy=edge_policy,
        )
        replay_path = output_root / f"replay_{name}.json"
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        evaluation = evaluate_posthoc_shadow(
            report_path=report_path_obj, shadow_result_path=replay_path
        )
        best = evaluation["hypotheses"][0]
        support = []
        for item in replay.get("plug_support_diagnostics", []):
            support.extend(item if isinstance(item, list) else [item])
        support_summary = {
            "min_visible_depth_support": min(
                (x["visible_depth_support_fraction"] for x in support),
                default=0.0,
            ),
            "max_foreground_occluded": max(
                (x["foreground_occluded_fraction"] for x in support),
                default=0.0,
            ),
            "min_in_frame": min(
                (x["in_frame_fraction"] for x in support), default=0.0
            ),
        }
        summary[name] = {
            "status": replay["status"],
            "success": replay["success"],
            "pose_valid": replay["pose_valid"],
            "residual_rms": best.get("residual_rms"),
            "condition_number": best.get("condition_number"),
            "translation_error_mm": best["posthoc_error"]["translation_error_m"]
            * 1000.0,
            "rotation_error_deg": best["posthoc_error"][
                "rotation_geodesic_error_deg"
            ],
            "support": support_summary,
            "replay_path": str(replay_path),
        }
    ab = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "groups": summary,
        "decision": "NO_CANDIDATE_ACCEPTED",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    (output_root / "occlusion_ab_summary.json").write_text(
        json.dumps(ab, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return ab


def objective_decomposition(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    replay_path: Path | str,
) -> dict[str, Any]:
    """Evaluate the same formal objective at nominal, optimizer, and actual C2."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    plug_cad, receptacle_cad = proxy_cad_points()
    nominal = np.asarray(report["posthoc_t_hand_plug_nominal"])
    actual = np.asarray(report["posthoc_t_hand_plug_actual"])
    optimizer_hp = np.asarray(
        replay["c2"]["hypotheses"][0]["T_hand_plug_xyz_rpy"]
    )
    states = {
        "nominal": np.concatenate((matrix_pose(nominal), np.zeros(6))),
        "optimizer": np.concatenate((optimizer_hp, np.zeros(6))),
        "posthoc_actual_yaw0": np.concatenate((matrix_pose(actual), np.zeros(6))),
        "posthoc_actual_yaw_pi": np.concatenate(
            (c2_action_pose6(matrix_pose(actual)), np.zeros(6))
        ),
    }
    frozen = tuple(False for _ in range(6)) + tuple(True for _ in range(6))
    group_names = ("edge", "depth", "normal", "support", "occlusion")
    decompositions = {}
    for state_name, state in states.items():
        problem = _ResidualProblem(
            views,
            plug_cad,
            receptacle_cad,
            state,
            frozen_mask=frozen,
            endpoints=("plug",),
        )
        residual = problem.residual(state)
        sizes = list(problem.group_sizes)
        view_groups = []
        cursor = 0
        view_count = len(views)
        group_len = sizes[0]
        for view_index in range(view_count):
            groups = {}
            for group_name in group_names:
                chunk = residual[cursor:cursor + group_len]
                groups[group_name] = {
                    "sum_squares": float(np.sum(chunk ** 2)),
                    "rms": float(np.sqrt(np.mean(chunk ** 2))),
                }
                cursor += group_len
            view_groups.append(
                {"view_id": views[view_index].view_id, "groups": groups}
            )
        prior = residual[cursor:]
        decompositions[state_name] = {
            "cost_sum_squares": float(np.sum(residual ** 2)),
            "prior_sum_squares": float(np.sum(prior ** 2)),
            "views": view_groups,
        }
    nominal_cost = decompositions["nominal"]["cost_sum_squares"]
    actual_cost = min(
        decompositions["posthoc_actual_yaw0"]["cost_sum_squares"],
        decompositions["posthoc_actual_yaw_pi"]["cost_sum_squares"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "objective": decompositions,
        "nominal_to_actual_cost_delta": actual_cost - nominal_cost,
        "objective_consistent_with_posthoc_truth": actual_cost < nominal_cost,
        "decision": (
            "OBJECTIVE_CONSISTENT_WITH_POSTHOC_TRUTH"
            if actual_cost < nominal_cost
            else "OBJECTIVE_INCONSISTENT_WITH_POSTHOC_TRUTH"
        ),
    }
    return result


def numerical_gradient_diagnostic(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
) -> dict[str, Any]:
    """Finite-difference sensitivity of the data objective at nominal."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    plug_cad, receptacle_cad = proxy_cad_points()
    nominal = np.asarray(report["posthoc_t_hand_plug_nominal"])
    state = np.concatenate((matrix_pose(nominal), np.zeros(6)))
    frozen = tuple(False for _ in range(6)) + tuple(True for _ in range(6))
    problem = _ResidualProblem(
        views,
        plug_cad,
        receptacle_cad,
        state,
        frozen_mask=frozen,
        include_prior=False,
        endpoints=("plug",),
    )
    base = float(np.sum(problem.residual(state) ** 2))
    names = ("tx", "ty", "tz", "rx", "ry", "rz")
    steps = (1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4)
    rows = []
    for index, (name, step) in enumerate(zip(names, steps)):
        plus = state.copy()
        plus[index] += step
        minus = state.copy()
        minus[index] -= step
        cost_plus = float(np.sum(problem.residual(plus) ** 2))
        cost_minus = float(np.sum(problem.residual(minus) ** 2))
        rows.append(
            {
                "parameter": name,
                "step": step,
                "cost_plus": cost_plus,
                "cost_minus": cost_minus,
                "central_gradient": (cost_plus - cost_minus) / (2.0 * step),
                "delta_cost_plus": cost_plus - base,
                "delta_cost_minus": cost_minus - base,
            }
        )
    flat = [row for row in rows if abs(row["delta_cost_plus"]) < 1.0e-9 and abs(row["delta_cost_minus"]) < 1.0e-9]
    return {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "base_cost_sum_squares": base,
        "parameters": rows,
        "quantized_flat_parameters": [row["parameter"] for row in flat],
        "decision": (
            "QUANTIZED_FLAT_DETECTED" if flat else "FINITE_SENSITIVITY_PRESENT"
        ),
    }


def run_optimizer_variant_ab(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """A0 default / B2a physical central Jacobian / B2b deterministic multistart."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate(
        (matrix_pose(nominal_hp), np.zeros(6, dtype=np.float64))
    )
    variants = [
        ("A0_baseline", "baseline", 0),
        ("B2a_physical_jacobian", "physical_jacobian", 0),
        ("B2b_multistart_physical_jacobian", "multistart_physical_jacobian", 9),
    ]
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path_obj = Path(report_path)
    summary = {}
    for name, variant, starts in variants:
        replay = estimate_postgrasp_T_HP(
            views,
            initial,
            optimizer_variant=variant,
            multistart_count=starts,
        )
        replay_path = output_root / f"replay_{name}.json"
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        evaluation = evaluate_posthoc_shadow(
            report_path=report_path_obj, shadow_result_path=replay_path
        )
        best = evaluation["hypotheses"][0]
        hypothesis = replay["c2"]["hypotheses"][0]
        summary[name] = {
            "status": replay["status"],
            "success": replay["success"],
            "pose_valid": replay["pose_valid"],
            "objective_cost": hypothesis.get("cost"),
            "residual_rms": best.get("residual_rms"),
            "condition_number": best.get("condition_number"),
            "translation_error_mm": best["posthoc_error"]["translation_error_m"]
            * 1000.0,
            "rotation_error_deg": best["posthoc_error"][
                "rotation_geodesic_error_deg"
            ],
            "solver_status": hypothesis.get("solver_status"),
            "solver_message": hypothesis.get("solver_message"),
            "solver_nfev": hypothesis.get("solver_nfev"),
            "solver_njev": hypothesis.get("solver_njev"),
            "solver_optimality": hypothesis.get("solver_optimality"),
            "active_mask": hypothesis.get("solver_active_mask"),
            "multistart_count": replay.get("multistart_count", 0),
            "replay_path": str(replay_path),
        }
    ab = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "variants": summary,
        "decision": "NO_VARIANT_ACCEPTED",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    (output_root / "optimizer_ab_summary.json").write_text(
        json.dumps(ab, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return ab


def jacobian_comparison_diagnostic(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
) -> dict[str, Any]:
    """Compare default scipy Jacobian with physical central Jacobian."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    plug_cad, receptacle_cad = proxy_cad_points()
    nominal = np.asarray(report["posthoc_t_hand_plug_nominal"])
    state = np.concatenate((matrix_pose(nominal), np.zeros(6)))
    frozen = tuple(False for _ in range(6)) + tuple(True for _ in range(6))
    result = {}
    for mode in ("default", "physical_central"):
        problem = _ResidualProblem(
            views, plug_cad, receptacle_cad, state,
            frozen_mask=frozen, endpoints=("plug",),
        )
        solved = problem.solve(jacobian_mode=mode)
        jac = np.asarray(solved["jacobian"], dtype=np.float64)
        singular = np.linalg.svd(jac, compute_uv=False)
        positive = singular[singular > 1.0e-12]
        result[mode] = {
            "column_norms": np.linalg.norm(jac, axis=0).tolist(),
            "singular_values": singular.tolist(),
            "rank_gt_1e-12": int(np.sum(positive > 0)),
            "condition_number": float(positive[0] / positive[-1]) if positive.size else None,
            "solver_status": solved["solver_status"],
            "solver_message": solved["solver_message"],
            "solver_nfev": solved["solver_nfev"],
            "solver_njev": solved["solver_njev"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "jacobian_comparison": result,
        "decision": "PHYSICAL_JACOBIAN_AVAILABLE",
    }


def run_cad_profile_ab(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """A0 legacy axisymmetric CAD vs B3 Shell25J C2-visible CAD."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate(
        (matrix_pose(nominal_hp), np.zeros(6, dtype=np.float64))
    )
    variants = [
        ("A0_legacy_axisymmetric", "legacy_axisymmetric"),
        ("B3_shell25j_c2_visible", "shell25j_c2_visible"),
    ]
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path_obj = Path(report_path)
    summary = {}
    for name, profile in variants:
        replay = estimate_postgrasp_T_HP(views, initial, cad_profile=profile)
        replay_path = output_root / f"replay_{name}.json"
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        evaluation = evaluate_posthoc_shadow(
            report_path=report_path_obj, shadow_result_path=replay_path
        )
        best = evaluation["hypotheses"][0]
        error = best["posthoc_error"]
        summary[name] = {
            "status": replay["status"],
            "success": replay["success"],
            "pose_valid": replay["pose_valid"],
            "objective_cost": replay["c2"]["hypotheses"][0]["cost"],
            "residual_rms": best.get("residual_rms"),
            "condition_number": best.get("condition_number"),
            "translation_error_mm": error["translation_error_m"] * 1000.0,
            "axis_tilt_deg": math.degrees(error["axis_tilt_rad"]),
            "local_rz_mod_pi_deg": math.degrees(error["local_rz_mod_pi_rad"]),
            "c2_modulo_rotation_deg": math.degrees(
                error["c2_modulo_rotation_rad"]
            ),
            "profile_id": replay.get("cad_profile"),
            "profile_metadata": replay.get("cad_profile_metadata"),
            "nfev": replay["c2"]["hypotheses"][0].get("solver_nfev"),
            "replay_path": str(replay_path),
        }
    ab = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "variants": summary,
        "decision": "NO_PROFILE_ACCEPTED",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    (output_root / "cad_profile_ab_summary.json").write_text(
        json.dumps(ab, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return ab


def run_feature_ablation_ab(
    *,
    report_path: Path | str,
    formal_archive_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Per-view/per-feature Shell25J C2 ablation on the dual-camera archive."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    views = load_formal_archive(formal_archive_path)
    nominal_hp = np.asarray(
        report["posthoc_t_hand_plug_nominal"], dtype=np.float64
    )
    initial = np.concatenate((matrix_pose(nominal_hp), np.zeros(6)))
    wrist_views = [v for v in views if v.group == "postgrasp_inhand_views"]
    fixed_views = [v for v in views if v.group == "fixed_world_camera_views"]
    groups = {
        "wrist_v0": wrist_views,
        "fixed_v0": fixed_views,
        "dual": views,
    }
    # A0 legacy baseline on dual views.
    legacy_plug, legacy_rec = proxy_cad_points()
    baseline = estimate_postgrasp_T_HP(views, initial)
    baseline_eval = evaluate_posthoc_shadow(
        report_path=Path(report_path), shadow_result_path=None
    ) if False else None
    baseline_path = Path(output_path) / "replay_A0_legacy_dual.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(baseline, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8"
    )
    baseline_eval = evaluate_posthoc_shadow(
        report_path=Path(report_path), shadow_result_path=baseline_path
    )
    base_best = baseline_eval["hypotheses"][0]["posthoc_error"]
    base_metric = {
        "trans": base_best["translation_error_m"],
        "tilt": base_best["axis_tilt_rad"],
        "rz": base_best["local_rz_mod_pi_rad"],
        "c2": base_best["c2_modulo_rotation_rad"],
        "objective": baseline["c2"]["hypotheses"][0]["cost"],
    }
    feature_sets = ("shell_only", "socket_only", "shell_plus_socket")
    summaries = {}
    accepted = {}
    for feature in feature_sets:
        from kcg_connector.d38999_cad_registration import shell25j_plug_cad_profile

        profile = shell25j_plug_cad_profile(feature_set=feature)
        for group_name, group_views in groups.items():
            if not group_views:
                continue
            replay = estimate_postgrasp_T_HP(
                group_views,
                initial,
                plug_cad=profile.plug_mating,
                receptacle_cad=profile.receptacle,
                cad_profile=f"shell25j_{feature}",
            )
            replay_path = Path(output_path) / f"replay_{feature}_{group_name}.json"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_path.write_text(
                json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
                + "\n", encoding="utf-8"
            )
            evaluation = evaluate_posthoc_shadow(
                report_path=Path(report_path), shadow_result_path=replay_path
            )
            best = evaluation["hypotheses"][0]
            error = best["posthoc_error"]
            # Truth-free local Rz cost curve around nominal.
            rz_curve = {}
            for delta_deg in (1.0, 2.0, 3.0):
                delta_rad = math.radians(delta_deg)
                for sign, label in ((1.0, f"p{delta_deg:g}"), (-1.0, f"m{delta_deg:g}")):
                    hp = matrix_pose(nominal_hp).copy()
                    hp[5] += sign * delta_rad
                    state = np.concatenate((hp, np.zeros(6)))
                    problem = _ResidualProblem(
                        group_views,
                        profile.plug_mating,
                        profile.receptacle,
                        state,
                        frozen_mask=(False,) * 6 + (True,) * 6,
                        endpoints=("plug",),
                        include_prior=False,
                    )
                    rz_curve[label] = float(np.sum(problem.residual(state) ** 2))
            support = {}
            for diagnostics in replay.get("plug_support_diagnostics", []):
                for item in diagnostics:
                    support[item["view_id"]] = item
            solved_hp = np.asarray(
                replay["c2"]["hypotheses"][0]["T_hand_plug_xyz_rpy"]
            )
            solved_state = np.concatenate((solved_hp, np.zeros(6)))
            loss_problem = _ResidualProblem(
                group_views,
                profile.plug_mating,
                profile.receptacle,
                solved_state,
                frozen_mask=(False,) * 6 + (True,) * 6,
                endpoints=("plug",),
                include_prior=False,
            )
            loss_residual = loss_problem.residual(solved_state)
            loss_sizes = list(loss_problem.group_sizes)
            loss_names = ("edge", "depth", "normal", "support", "occlusion")
            per_view_loss = []
            cursor = 0
            for view in group_views:
                row = {"view_id": view.view_id}
                for loss_name in loss_names:
                    chunk = loss_residual[cursor:cursor + loss_sizes[0]]
                    row[f"{loss_name}_sum_squares"] = float(np.sum(chunk ** 2))
                    cursor += loss_sizes[0]
                per_view_loss.append(row)
            summary = {
                "status": replay["status"],
                "success": replay["success"],
                "pose_valid": replay["pose_valid"],
                "objective": replay["c2"]["hypotheses"][0]["cost"],
                "residual_rms": best.get("residual_rms"),
                "condition_number": best.get("condition_number"),
                "translation_error_mm": error["translation_error_m"] * 1000.0,
                "axis_tilt_deg": math.degrees(error["axis_tilt_rad"]),
                "local_rz_mod_pi_deg": math.degrees(error["local_rz_mod_pi_rad"]),
                "c2_modulo_rotation_deg": math.degrees(
                    error["c2_modulo_rotation_rad"]
                ),
                "local_rz_cost_curve": rz_curve,
                "per_view_support": support,
                "per_view_loss_cost": per_view_loss,
                "replay_path": str(replay_path),
            }
            summaries[f"{feature}_{group_name}"] = summary
            if group_name == "dual":
                flags = {
                    "translation_improved": error["translation_error_m"] <= base_metric["trans"] + 1.0e-12,
                    "axis_tilt_improved": error["axis_tilt_rad"] <= base_metric["tilt"] + 1.0e-12,
                    "local_rz_improved": error["local_rz_mod_pi_rad"] <= base_metric["rz"] + 1.0e-12,
                    "c2_modulo_improved": error["c2_modulo_rotation_rad"] <= base_metric["c2"] + 1.0e-12,
                    "objective_not_worse": replay["c2"]["hypotheses"][0]["cost"] <= base_metric["objective"] + 1.0e-9,
                }
                accepted[feature] = flags
                summary.update({"acceptance_flags": flags})
    decision = (
        "CANDIDATE_ACCEPTED_FOR_NEW_SEED_GPU"
        if any(all(flags.values()) for flags in accepted.values())
        else "NO_CANDIDATE_ACCEPTED"
    )
    ab = {
        "schema_version": SCHEMA_VERSION,
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "baseline": base_metric,
        "variants": summaries,
        "decision": decision,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    ab_path = Path(output_path) / "feature_ablation_summary.json"
    ab_path.write_text(
        json.dumps(ab, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8"
    )
    return ab


def _default_repository():
    return Path(__file__).resolve().parents[3]


def main() -> int:
    repository = _default_repository()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_codex_shadow_smoke_dual_camera_v0/seed000"
            / "nominal_physics_report.json"
        ),
    )
    parser.add_argument(
        "--shadow-result",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_codex_shadow_smoke_dual_camera_v0/seed000"
            / "postgrasp_shadow/shadow_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/offline_posthoc_shadow_evaluation.json"
        ),
    )
    parser.add_argument("--replay-archive", default=None)
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--run-view-ab", action="store_true")
    parser.add_argument("--run-occlusion-ab", action="store_true")
    parser.add_argument("--run-optimizer-ab", action="store_true")
    parser.add_argument("--run-cad-profile-ab", action="store_true")
    parser.add_argument("--run-feature-ablation-ab", action="store_true")
    parser.add_argument("--objective-diagnostic", action="store_true")
    parser.add_argument("--gradient-diagnostic", action="store_true")
    parser.add_argument("--jacobian-comparison", action="store_true")
    parser.add_argument("--view-ab-output", default=None)
    parser.add_argument(
        "--plug-feature-set",
        choices=("mating_only", "mating_plus_nut_body"),
        default="mating_only",
    )
    args = parser.parse_args()
    if args.objective_diagnostic:
        result = objective_decomposition(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            replay_path=args.shadow_result,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if args.jacobian_comparison:
        result = jacobian_comparison_diagnostic(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if args.gradient_diagnostic:
        result = numerical_gradient_diagnostic(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if args.run_feature_ablation_ab:
        ab = run_feature_ablation_ab(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            output_path=args.view_ab_output
            or str(
                _default_repository()
                / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
                / "deepseek/offline_feature_ablation_ab"
            ),
        )
        print(json.dumps(ab, sort_keys=True, indent=2))
        return 0
    if args.run_cad_profile_ab:
        ab = run_cad_profile_ab(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            output_path=args.view_ab_output
            or str(
                _default_repository()
                / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
                / "deepseek/offline_cad_profile_ab"
            ),
        )
        print(json.dumps(ab, sort_keys=True, indent=2))
        return 0
    if args.run_optimizer_ab:
        ab = run_optimizer_variant_ab(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            output_path=args.view_ab_output
            or str(
                _default_repository()
                / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
                / "deepseek/offline_optimizer_ab"
            ),
        )
        print(json.dumps(ab, sort_keys=True, indent=2))
        return 0
    if args.run_occlusion_ab:
        ab = run_occlusion_policy_ab(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            output_path=args.view_ab_output
            or str(
                _default_repository()
                / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
                / "deepseek/offline_occlusion_ab"
            ),
        )
        print(json.dumps(ab, sort_keys=True, indent=2))
        return 0
    if args.run_view_ab:
        ab = run_T_HP_view_ab(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            output_path=args.view_ab_output
            or str(
                _default_repository()
                / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
                / "deepseek/offline_view_ab"
            ),
        )
        print(json.dumps(ab, sort_keys=True, indent=2))
        return 0
    if args.replay_archive:
        replay = replay_shadow_estimate(
            report_path=args.report,
            formal_archive_path=args.replay_archive,
            plug_feature_set=args.plug_feature_set,
        )
        replay_path = Path(args.replay_output or args.output)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        result = evaluate_posthoc_shadow(
            report_path=args.report,
            shadow_result_path=replay_path,
        )
        result["replayed_shadow_path"] = str(replay_path)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    result = evaluate_posthoc_shadow(
        report_path=args.report,
        shadow_result_path=args.shadow_result,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
