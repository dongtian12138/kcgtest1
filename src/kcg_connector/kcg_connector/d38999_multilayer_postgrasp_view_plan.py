"""Evidence-bound static post-grasp view plan for the multilayer model.

The existing V0/V1/V2 deltas are retained as precommitted candidates.  This
module deliberately does not turn them into executable or accepted views:
legacy evidence rejects V1 at the joint-motion budget and V2 at IK, B4 has no
dynamic pass, and the current wrist camera remains layout-evidence-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_postgrasp_view_plan_v1"

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml": (
        "cf2f0c75bf8f128f5d52057d60841697df8a823dfa1507e3d05f49573ef470e9"
    ),
    "src/kcg_connector/kcg_connector/postgrasp_shadow_view_planner.py": (
        "e749fb0893043351f514323413680a021d9b407b06333b4c939c76865b4ad5bb"
    ),
    "src/kcg_connector/isaac/postgrasp_shadow_capture_runtime.py": (
        "281aa6bf2967e4a593b8120a103f307434a557651574c6686fee8d673f1bae5e"
    ),
    "src/kcg_connector/test/test_postgrasp_rework_contracts.py": (
        "3437bc329052c84ddb8cb822df77e197616edeef84d75e1a45e00bb43de2084e"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C1-PALM-CAMERA-INTERFACE/"
    "INTERFACE_MANIFEST.json": (
        "b1d5a8938a918898302a7b4790025dc94a97abbdb844fb020db195a0b160d990"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C2-WRIST-CAMERA-INTERFACE/"
    "INTERFACE_MANIFEST.json": (
        "09b4369e08a2556fbc35fb0e1ec4d07f7376a80982bbfb049f5eb5c481fe506f"
    ),
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json": (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
}

EXPECTED_DELTAS = {
    "V0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "V1": [
        0.012,
        -0.006,
        -0.030,
        0.06981317007977318,
        -0.17453292519943295,
        0.0,
    ],
    "V2": [
        -0.012,
        0.006,
        -0.030,
        -0.06981317007977318,
        0.17453292519943295,
        0.0,
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verified_sources(root: Path) -> tuple[dict[str, str], ...]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen postgrasp source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen postgrasp source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_postgrasp_view_plan(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Return the current static plan without authorizing motion or capture."""

    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    contract = _mapping(
        root / "src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml",
        "postgrasp view contract",
    )
    palm = _mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C1-PALM-CAMERA-INTERFACE/INTERFACE_MANIFEST.json",
        "Palm interface manifest",
    )
    wrist = _mapping(
        root
        / "artifacts/agent_control/tasks/"
        "EIGHT-HOUR-C2-WRIST-CAMERA-INTERFACE/INTERFACE_MANIFEST.json",
        "wrist interface manifest",
    )
    model_mapping = _mapping(
        root
        / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "MODEL_MAPPING.json",
        "multilayer model mapping",
    )

    if (
        contract.get("schema_version") != "kcg_d38999_postgrasp_shadow_v1"
        or contract.get("threshold_label") != "SIM_TUNING_ONLY_CANDIDATE"
    ):
        raise ValueError("postgrasp contract identity changed")
    motion = contract.get("motion")
    views = contract.get("views")
    inputs = contract.get("formal_inputs")
    if not all(isinstance(value, Mapping) for value in (motion, views, inputs)):
        raise ValueError("postgrasp motion, views, or input firewall missing")
    expected_motion = {
        "move_duration_s": 2.0,
        "settle_duration_s": 0.5,
        "per_command_max_joint_delta_rad": 0.05,
        "planned_max_joint_inf_rad": 0.05,
        "episode_max_joint_inf_rad": 0.20,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    if dict(motion) != expected_motion:
        raise ValueError("postgrasp motion limits changed")
    if (
        views.get("postgrasp_reference") != "V0"
        or views.get("postgrasp_inhand_views") != ["V0", "V1"]
        or views.get("optional_third_view") != "V2"
    ):
        raise ValueError("postgrasp candidate sequence changed")
    deltas = views.get("predefined_tcp_deltas")
    if not isinstance(deltas, Mapping) or any(
        deltas.get(view_id) != value
        for view_id, value in EXPECTED_DELTAS.items()
    ):
        raise ValueError("postgrasp candidate deltas changed")
    limits = views.get("candidate_limits")
    if not isinstance(limits, Mapping) or dict(limits) != {
        "maximum_translation_m": 0.040,
        "maximum_rotation_rad": 0.20943951023931953,
        "optical_axis_min_deg": 25.0,
        "optical_axis_max_deg": 70.0,
        "pairwise_optical_axis_min_deg": 20.0,
        "minimum_depth_valid_fraction": 0.05,
        "minimum_central_depth_fraction": 0.02,
    }:
        raise ValueError("postgrasp candidate limits changed")

    mount = views.get("wrist_camera_mount")
    if not isinstance(mount, Mapping):
        raise ValueError("legacy wrist mount missing")
    if (
        mount.get("eye_handbase_m") != wrist.get("v5_eye_handbase_m")
        or mount.get("target_handbase_m") != wrist.get("v5_target_handbase_m")
        or mount.get("control_authorized") is not False
    ):
        raise ValueError("legacy wrist numeric mount or authorization changed")

    if (
        palm.get("status") != "OFFLINE_PASS"
        or palm.get("interface_role") != "PALM_RGBD_SHADOW_CAPABLE"
        or palm.get("dynamic_camera_pass_claimed") is not False
        or wrist.get("status") != "OFFLINE_PASS"
        or wrist.get("interface_role") != "WRIST_LAYOUT_EVIDENCE_ONLY"
        or wrist.get("dynamic_camera_pass_claimed") is not False
    ):
        raise ValueError("current camera interface evidence is not fail-closed")
    wrist_policy = wrist.get("layout_evidence_policy")
    if not isinstance(wrist_policy, Mapping) or any(
        wrist_policy.get(key) is not False
        for key in (
            "shadow_inference_allowed",
            "control_input_allowed",
            "selected_for_control_allowed",
        )
    ):
        raise ValueError("current wrist role was promoted")
    if palm.get("observation_target") != wrist.get("observation_target"):
        raise ValueError("Palm and wrist observation targets differ")
    try:
        visual = model_mapping["representations"]["D38999_VISUAL_COMPLETE_V1"]
    except (KeyError, TypeError):
        raise ValueError("multilayer visual mapping missing") from None
    if (
        visual.get("root") != palm["observation_target"]["root"]
        or visual.get("pair_root") != palm["observation_target"]["pair_root"]
        or visual.get("visible_geometry_preserved") is not True
    ):
        raise ValueError("multilayer visual target changed")

    allowed_inputs = set(inputs.get("allowed_fields", []))
    forbidden_inputs = set(inputs.get("forbidden_fields", []))
    if not {
        "view_id",
        "timestamp_utc",
        "rgb",
        "depth",
        "camera_intrinsics",
        "actual_arm_q_rad",
        "T_WH_from_fk",
    }.issubset(allowed_inputs):
        raise ValueError("postgrasp allowed image/FK inputs changed")
    if not {
        "semantic",
        "registered_truth_xy_m",
        "object_truth",
        "contact_report",
        "collider_identity",
    }.issubset(forbidden_inputs):
        raise ValueError("postgrasp truth firewall weakened")

    evidence_by_view = {
        "V0": {
            "legacy_evidence": "H0_FIXED_WORLD_CAMERA_CAPTURE_PATH_PRESENT",
            "known_legacy_seed0_gate": "CAPTURE_PATH_ONLY",
        },
        "V1": {
            "legacy_evidence": "SEED0_PLANNED_JOINT_BUDGET_REJECTED",
            "known_legacy_seed0_gate": "PLANNED_MAX_JOINT_INF_RAD_EXCEEDED",
        },
        "V2": {
            "legacy_evidence": "SEED0_IK_FAIL_CLOSED",
            "known_legacy_seed0_gate": "IK_FAILURE",
        },
    }
    candidate_views = []
    for index, view_id in enumerate(("V0", "V1", "V2")):
        candidate_views.append(
            {
                "sequence_index": index,
                "view_id": view_id,
                "required_by_legacy_contract": view_id in {"V0", "V1"},
                "optional_third_view": view_id == "V2",
                "tcp_delta_xyz_rpy": list(EXPECTED_DELTAS[view_id]),
                **evidence_by_view[view_id],
                "current_multilayer_execution_proven": False,
                "dynamic_execution_authorized": False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "STATIC_PASS",
        "dynamic_status": "PARKED",
        "classification": "PRECOMMITTED_VIEW_CANDIDATES_STATIC_ONLY",
        "plan_role": "POSTGRASP_VIEW_CANDIDATE_INTERFACE",
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
        "candidate_view_count": 3,
        "candidate_sequence": ["V0", "V1", "V2"],
        "candidate_views": candidate_views,
        "motion_limits": expected_motion,
        "candidate_limits": dict(limits),
        "camera_binding": {
            "legacy_config_mount_role": "wrist_camera_mount",
            "legacy_mount_contract_label": mount["contract"],
            "current_wrist_role": wrist["interface_role"],
            "current_palm_role": palm["interface_role"],
            "selected_camera_for_dynamic_capture": None,
            "selection_reason": (
                "WRIST_LAYOUT_ONLY_AND_NO_CURRENT_MULTILAYER_DYNAMIC_CAMERA_PASS"
            ),
            "observation_target": dict(palm["observation_target"]),
        },
        "view_independence_semantics": {
            "hand_camera_and_grasped_plug_comoving": True,
            "arm_motion_adds_independent_T_HP_view": False,
            "fixed_world_camera_can_add_independent_T_HP_view": True,
        },
        "dynamic_readiness": {
            "current_multilayer_dynamic_views_proven": 0,
            "legacy_h0_capture_path_present": True,
            "legacy_v1_joint_budget_rejected": True,
            "legacy_v2_ik_failed": True,
            "formal_capture_dependency": "B4_DYNAMIC_PASS",
            "b4_dynamic_pass_evidence_present": False,
            "formal_capture_authorized": False,
            "dynamic_view_plan_pass_claimed": False,
        },
        "truth_firewall": {
            "allowed_fields": sorted(allowed_inputs),
            "forbidden_fields": sorted(forbidden_inputs),
            "runtime_image_changes_precommitted_motion": False,
            "object_pose_truth_changes_motion": False,
            "contact_name_or_normal_changes_motion": False,
            "postrun_object_pose_write_allowed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "robot_motion_started": False,
        "render_capture_performed": False,
        "dynamic_visual_pass_claimed": False,
        "formal_postgrasp_capture_pass_claimed": False,
        "hardware_authorized": False,
    }
