"""CPU-only POSTHOC_TRUTH_ONLY visibility A/B contract.

Without explicit posthoc frame data (T_WR, FK targets, frozen T_HC) this tool
must output INSUFFICIENT_POSTHOC_FRAME_DATA and must never fabricate an
A_FEASIBLE decision from a world-identity receptacle frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kcg_connector.postgrasp_shadow_view_planner import (
    evaluate_visibility_ab_from_frames,
    write_ab_report,
)


def _repository():
    return Path(__file__).resolve().parents[3]


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    repository = _repository()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase0_codex_smoke/seed000/nominal_physics_report.json"
        ),
    )
    parser.add_argument(
        "--controller-steps",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase0_codex_smoke/seed000/controller_steps.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    parser.add_argument("--fixed-receptacle-pose-json", default=None)
    parser.add_argument("--view-fk-targets-json", default=None)
    parser.add_argument(
        "--output",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/offline_visibility_ab_phase0_seed000.json"
        ),
    )
    args = parser.parse_args()
    report = _load_json(args.report)
    if report.get("passed") is not True:
        raise RuntimeError("phase0 report did not pass")

    if not args.fixed_receptacle_pose_json or not args.view_fk_targets_json:
        insufficient = {
            "schema_version": "kcg_d38999_postgrasp_visibility_ab_v1",
            "truth_scope": "POSTHOC_TRUTH_ONLY",
            "decision": "INSUFFICIENT_POSTHOC_FRAME_DATA",
            "reason": (
                "requires explicit posthoc T_WR and actual view FK targets; "
                "world-identity receptacle frame and raw TCP deltas are not valid"
            ),
            "truth_used_for_control": False,
            "truth_used_for_estimator": False,
        }
        write_ab_report(Path(args.output), insufficient)
        print(json.dumps(insufficient, sort_keys=True, indent=2))
        return 0

    t_wr_doc = _load_json(args.fixed_receptacle_pose_json)
    view_doc = _load_json(args.view_fk_targets_json)
    t_wr = np.asarray(t_wr_doc["T_WR"], dtype=np.float64)
    t_hc = np.asarray(view_doc["T_HC_calibrated"], dtype=np.float64)
    postgrasp_hand_poses = [
        np.asarray(item["T_WH"], dtype=np.float64)
        for item in view_doc["postgrasp_hand_poses"]
    ]
    preinsert_hand_poses = [
        np.asarray(item["T_WH"], dtype=np.float64)
        for item in view_doc["preinsert_hand_poses"]
    ]
    t_hp = np.asarray(report["posthoc_t_hand_plug_actual"], dtype=np.float64)
    result = evaluate_visibility_ab_from_frames(
        postgrasp_hand_poses=postgrasp_hand_poses,
        preinsert_hand_poses=preinsert_hand_poses,
        t_hc=t_hc,
        t_wr=t_wr,
        t_hp=t_hp,
    )
    write_ab_report(Path(args.output), result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
