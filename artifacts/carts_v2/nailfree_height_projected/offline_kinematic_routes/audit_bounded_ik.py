#!/usr/bin/env python3
"""Audit Top-3 bounded IK waypoints without claiming path collision safety."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--offline-result", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    sys.path.insert(0, str(root / "src/kcg_connector/isaac/carts_v2"))
    from controller import build_joint_motion_plan
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    from kcg_connector.grasp.robust.bounded_hand_base_ik import (
        CandidateJointRouteError,
    )
    from kcg_connector.grasp.robust.object_model import file_sha256

    config = Path(args.config).resolve()
    result_path = Path(args.offline_result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("object_id") != args.object_id:
        raise ValueError("offline result object identity mismatch")
    if result.get("config_sha256") != file_sha256(config):
        raise ValueError("offline result config hash mismatch")
    if result.get("hardware_authorized") is not False:
        raise ValueError("hardware authorization changed")
    inputs = load_v2_inputs(root, config_path=config, object_id=args.object_id)
    formal_ids = {
        row["candidate_id"] for row in result.get("formal_task_candidates", [])
    }
    rows = []
    for candidate in result.get("research_task_candidates", [])[:3]:
        row = {
            "candidate_id": candidate["candidate_id"],
            "rank": candidate["rank"],
            "formal_task_top3": candidate["candidate_id"] in formal_ids,
        }
        try:
            plan = build_joint_motion_plan(
                root,
                inputs,
                candidate["control_plan"],
                inputs.frozen_world_from_object,
            )
            row.update(
                status="OFFLINE_KINEMATIC_REACHABLE_50MM_WAYPOINTS",
                motion_plan=plan,
            )
        except CandidateJointRouteError as error:
            row.update(
                status="OFFLINE_KINEMATIC_IK_FAILED",
                error_code=error.code,
                error_detail=error.detail,
            )
        rows.append(row)
    output = {
        "schema_version": "nailfree_bounded_ik_audit_v1",
        "object_id": args.object_id,
        "source_result": str(result_path.relative_to(root)),
        "source_result_sha256": file_sha256(result_path),
        "config_sha256": file_sha256(config),
        "hardware_authorized": False,
        "research_dynamic_pass": False,
        "formal_dynamic_pass": False,
        "runtime_collision_binding_accepted": False,
        "complete_arm_path_collision_checked": False,
        "kinematic_reachable_count": sum(
            row["status"].startswith("OFFLINE_KINEMATIC_REACHABLE") for row in rows
        ),
        "candidate_routes": rows,
        "claim_scope": (
            "DISCRETE_BOUNDED_IK_APPROACH_AND_50MM_LIFT_WAYPOINTS_ONLY; "
            "NOT_FULL_ARM_COLLISION_OR_EXECUTABLE_OR_DYNAMIC_EVIDENCE"
        ),
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{args.object_id}: {output['kinematic_reachable_count']}/{len(rows)} bounded-IK routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
