"""Diagnostic C4/C6 evaluation with formal/posthoc file boundary.

The formal loader reads only a pre-extracted truth-free capture contract and
raw RGB-D files.  A diagnostic sidecar builder may read the old 014 archive and
is explicitly marked DIAGNOSTIC_RECONSTRUCTION_ONLY; its output is not formal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kcg_connector.d38999_cad_registration import (
    CameraModel,
    fixed_camera_model,
)
from kcg_connector.d38999_inhand_multiview import matrix_pose
from kcg_connector.postgrasp_shadow_estimator import (
    FormalArchiveError,
    FormalView,
    estimate_postgrasp_T_HP,
)
from kcg_connector.posthoc_shadow_evaluation import evaluate_posthoc_shadow

CONTRACT_SCHEMA = "kcg_d38999_diag_capture_contract_v1"
TRUTH_FREE_CAPTURE_SCOPE = "TRUTH_FREE_VIRTUAL_MOUNT_DIAGNOSTIC_ONLY"


def _cv_camera_pose_from_usd_row_xform(matrix_rows) -> np.ndarray:
    usd = np.asarray(matrix_rows, dtype=np.float64)
    if usd.shape != (4, 4) or not np.all(np.isfinite(usd)):
        raise FormalArchiveError("invalid USD camera transform")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = usd[:3, :3].T @ np.diag((1.0, -1.0, -1.0))
    pose[:3, 3] = usd[3, :3]
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise FormalArchiveError("USD camera rotation is not orthonormal")
    return pose


def build_truth_free_capture_sidecar(
    *,
    base_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Build a diagnostic contract from the new truth-free GPU manifest."""
    base = Path(base_path)
    manifest = json.loads(
        (base / "diag_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("semantic_present") is not False:
        raise FormalArchiveError("semantic data is forbidden")
    if manifest.get("control_authorized") is not False:
        raise FormalArchiveError("diagnostic capture cannot be authorized")
    required_hashes = {
        "postgrasp_diag_mount_search_sha256",
        "isaac_d38999_rgbd_runtime_sha256",
        "rgbd_config_sha256",
    }
    if not required_hashes <= set(manifest.get("source_hashes", {})):
        raise FormalArchiveError("capture source/config hashes missing")
    candidates = {}
    for source in manifest.get("records", []):
        candidate_id = source.get("candidate_id")
        if candidate_id not in {"C4", "C6"}:
            continue
        required = {
            "capture_camera",
            "capture_T_WH",
            "capture_T_HC",
            "capture_T_WC",
            "capture_usd_camera_xform_row_major",
            "eye_plug_m",
            "global_physics_step",
            "target_plug_m",
            "timestamp_utc",
        }
        missing = sorted(required - set(source))
        if missing:
            raise FormalArchiveError(
                f"{candidate_id} missing capture fields: {missing}"
            )
        camera_record = source["capture_camera"]
        intrinsics = np.asarray(
            camera_record.get("intrinsics"), dtype=np.float64
        )
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise FormalArchiveError(f"{candidate_id} intrinsics invalid")
        t_wh = np.asarray(source["capture_T_WH"], dtype=np.float64)
        t_wc = _cv_camera_pose_from_usd_row_xform(
            source["capture_usd_camera_xform_row_major"]
        )
        t_hc = np.linalg.inv(t_wh) @ t_wc
        camera_in_plug = fixed_camera_model(
            eye=source["eye_plug_m"],
            target=source["target_plug_m"],
            resolution=tuple(camera_record["resolution"]),
            focal_length_mm=float(camera_record["focal_length_mm"]),
            horizontal_aperture_mm=float(
                camera_record["horizontal_aperture_mm"]
            ),
        )
        design_t_pc = _camera_pose(camera_in_plug)
        design_t_hc = np.asarray(
            source.get("design_T_HC", source["capture_T_HC"]),
            dtype=np.float64,
        )
        nominal_hp = design_t_hc @ np.linalg.inv(design_t_pc)
        candidates[candidate_id] = {
            "view_id": f"DIAG_{candidate_id}",
            "timestamp_utc": source["timestamp_utc"],
            "physics_step": int(source["global_physics_step"]),
            "camera_model": {
                "width": int(camera_record["resolution"][0]),
                "height": int(camera_record["resolution"][1]),
                "fx": float(intrinsics[0, 0]),
                "fy": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
                "position_world": t_wc[:3, 3].tolist(),
                "world_to_camera": t_wc[:3, :3].T.tolist(),
            },
            "T_WH": t_wh.tolist(),
            "T_HC": t_hc.tolist(),
            "T_WC": t_wc.tolist(),
            "nominal_hp_xyz_rpy": matrix_pose(nominal_hp).tolist(),
            "raw_rgb": f"raw_{candidate_id.lower()}/rgb.png",
            "raw_depth": f"raw_{candidate_id.lower()}/depth_m.npy",
            "source": TRUTH_FREE_CAPTURE_SCOPE,
        }
    if set(candidates) != {"C4", "C6"}:
        raise FormalArchiveError("C4/C6 capture records incomplete")
    sidecar = {
        "schema_version": CONTRACT_SCHEMA,
        "scope": TRUTH_FREE_CAPTURE_SCOPE,
        "formal_estimator_input": False,
        "control_authorized": False,
        "mount_formally_accepted": False,
        "source_hashes": manifest["source_hashes"],
        "candidates": candidates,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sidecar, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def build_diagnostic_reconstruction_sidecar(
    *,
    base_path: Path | str,
    report_path: Path | str,
    controller_steps_path: Path | str,
    pick_config_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Read the old 014 archive once and emit a truth-free sidecar."""
    base = Path(base_path)
    manifest = json.loads(
        (base / "diag_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    last = None
    with Path(controller_steps_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise FormalArchiveError("controller steps empty")
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
        load_d38999_tabletop_pick_config,
    )

    pick = load_d38999_tabletop_pick_config(Path(pick_config_path))
    arm_q = np.asarray(last["arm_q_actual_rad"], dtype=np.float64)
    tcp = np.asarray(iiwa14_grasp_tcp_transform(tuple(arm_q)))
    tcp_from_hand = np.eye(4)
    tcp_from_hand[2, 3] = -float(pick.geometry_candidate.handbase_to_tcp_m)
    current_hand = tcp @ tcp_from_hand
    nominal_hp = np.asarray(report["posthoc_t_hand_plug_nominal"])
    candidates = {}
    for record in manifest["records"]:
        if record.get("capture_status") != "CAPTURED":
            continue
        candidate_id = record["candidate_id"]
        eye_plug = tuple(float(v) for v in record["eye_plug_m"])
        target_plug = tuple(float(v) for v in record["target_plug_m"])
        camera_in_plug = fixed_camera_model(
            eye=eye_plug, target=target_plug, resolution=(1280, 720)
        )
        t_pc = _camera_pose(camera_in_plug)
        t_hc = nominal_hp @ t_pc
        t_wc = current_hand @ t_hc
        eye_world = tuple(float(v) for v in t_wc[:3, 3])
        forward_world = t_wc[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
        target_world = tuple(
            float(v) for v in (np.asarray(t_wc[:3, 3]) + forward_world)
        )
        world_camera = fixed_camera_model(
            eye=eye_world, target=target_world, resolution=(1280, 720)
        )
        candidates[candidate_id] = {
            "view_id": f"DIAG_{candidate_id}",
            "timestamp_utc": record["timestamp_utc"],
            "physics_step": int(last.get("global_step", -1)),
            "candidate_id": candidate_id,
            "camera_model": _camera_model_record(world_camera),
            "T_WH": current_hand.tolist(),
            "T_HC": t_hc.tolist(),
            "T_WC": t_wc.tolist(),
            "nominal_hp_xyz_rpy": matrix_pose(nominal_hp).tolist(),
            "raw_rgb": f"raw_{candidate_id.lower()}/rgb.png",
            "raw_depth": f"raw_{candidate_id.lower()}/depth_m.npy",
            "source": "DIAGNOSTIC_RECONSTRUCTION_ONLY_FROM_014",
        }
    sidecar = {
        "schema_version": CONTRACT_SCHEMA,
        "scope": "DIAGNOSTIC_RECONSTRUCTION_ONLY",
        "formal_estimator_input": False,
        "control_authorized": False,
        "mount_formally_accepted": False,
        "candidates": candidates,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sidecar, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def _camera_pose(camera: CameraModel) -> np.ndarray:
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    pose = np.eye(4)
    pose[:3, :3] = rows.T
    pose[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return pose


def _camera_model_record(camera: CameraModel) -> dict[str, Any]:
    return {
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "position_world": [float(v) for v in camera.position_world],
        "world_to_camera": [
            [float(v) for v in row] for row in camera.world_to_camera
        ],
    }


def load_diag_formal_view(
    contract_path: Path | str,
    raw_root: Path | str,
    candidate_id: str,
) -> FormalView:
    """Truth-free formal loader.  It does not accept a report path."""
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise FormalArchiveError("unsupported diag capture contract")
    if contract.get("scope") not in {
        "DIAGNOSTIC_RECONSTRUCTION_ONLY",
        TRUTH_FREE_CAPTURE_SCOPE,
    }:
        raise FormalArchiveError("diag contract is not diagnostic-only")
    record = contract.get("candidates", {}).get(candidate_id)
    if record is None:
        raise FormalArchiveError(f"missing candidate contract: {candidate_id}")
    required = (
        "timestamp_utc",
        "physics_step",
        "camera_model",
        "T_WH",
        "T_HC",
        "T_WC",
        "nominal_hp_xyz_rpy",
        "raw_rgb",
        "raw_depth",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise FormalArchiveError(f"missing contract fields: {missing}")
    if int(record["physics_step"]) < 0:
        raise FormalArchiveError("physics_step invalid")
    cam = record["camera_model"]
    camera = CameraModel(
        width=int(cam["width"]),
        height=int(cam["height"]),
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
        position_world=tuple(cam["position_world"]),
        world_to_camera=tuple(tuple(row) for row in cam["world_to_camera"]),
    )
    rgb_bgr = cv2.imread(str(Path(raw_root) / record["raw_rgb"]))
    if rgb_bgr is None:
        raise FormalArchiveError(f"{candidate_id} rgb missing")
    depth_path = Path(raw_root) / record["raw_depth"]
    if not depth_path.is_file():
        raise FormalArchiveError(f"{candidate_id} depth missing")
    depth = np.load(depth_path).astype(np.float32)
    return FormalView(
        view_id=record["view_id"],
        timestamp_utc=record["timestamp_utc"],
        rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth=depth,
        camera=camera,
        T_WH=np.asarray(record["T_WH"]),
        T_WC=np.asarray(record["T_WC"]),
        T_HC=np.asarray(record["T_HC"]),
        group="postgrasp_inhand_views",
        extrinsic_source="DIAGNOSTIC_RECONSTRUCTION_ONLY",
    )


def run_formal_only(
    contract_path: Path | str,
    raw_root: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for candidate_id in ("C4", "C6"):
        nominal_hp = np.asarray(
            contract["candidates"][candidate_id]["nominal_hp_xyz_rpy"]
        )
        initial = np.concatenate((nominal_hp, np.zeros(6)))
        view = load_diag_formal_view(contract_path, raw_root, candidate_id)
        replay = estimate_postgrasp_T_HP(
            [view],
            initial,
            cad_profile="shell25j_c2_visible",
            cad_profile_feature_set="shell_plus_socket",
        )
        path = out / f"replay_{candidate_id}.json"
        path.write_text(
            json.dumps(replay, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        results[candidate_id] = {
            "path": str(path),
            "status": replay["status"],
            "success": replay["success"],
            "pose_valid": replay["pose_valid"],
        }
    return {
        "schema_version": "kcg_d38999_diag_formal_v1",
        "scope": contract["scope"],
        "control_authorized": False,
        "results": results,
    }


def run_posthoc_comparison(
    report_path: Path | str,
    formal_output_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    formal_root = Path(formal_output_path)
    baseline = {
        "wrist_v0": {
            "translation_error_mm": 12.8624,
            "axis_tilt_deg": 9.5439,
            "local_rz_mod_pi_deg": 3.5041,
            "condition_number": None,
        },
        "wrist_plus_fixed": {
            "translation_error_mm": 1.4078,
            "axis_tilt_deg": 0.2532,
            "local_rz_mod_pi_deg": 3.0869,
            "condition_number": 2.04e7,
        },
    }
    variants = {}
    for candidate_id in ("C4", "C6"):
        replay_path = formal_root / f"replay_{candidate_id}.json"
        evaluation = evaluate_posthoc_shadow(
            report_path=Path(report_path), shadow_result_path=replay_path
        )
        best = evaluation["hypotheses"][0]
        variants[candidate_id] = {
            "translation_error_mm": best["posthoc_error"][
                "translation_error_m"
            ]
            * 1000.0,
            "axis_tilt_deg": math.degrees(
                best["posthoc_error"]["axis_tilt_rad"]
            ),
            "local_rz_mod_pi_deg": math.degrees(
                best["posthoc_error"]["local_rz_mod_pi_rad"]
            ),
            "condition_number": best.get("condition_number"),
        }
    flags = {}
    for candidate_id, metrics in variants.items():
        candidate_flags = {}
        for baseline_name in baseline:
            candidate_flags[baseline_name] = {
                "translation_not_worse": metrics["translation_error_mm"]
                <= baseline[baseline_name]["translation_error_mm"] + 1.0e-12,
                "tilt_not_worse": metrics["axis_tilt_deg"]
                <= baseline[baseline_name]["axis_tilt_deg"] + 1.0e-12,
                "local_rz_not_worse": metrics["local_rz_mod_pi_deg"]
                <= baseline[baseline_name]["local_rz_mod_pi_deg"] + 1.0e-12,
                "condition_not_worse": (
                    baseline[baseline_name]["condition_number"] is None
                    or metrics["condition_number"] is None
                    or metrics["condition_number"]
                    <= baseline[baseline_name]["condition_number"] + 1.0e-12
                ),
            }
        candidate_flags["accepted"] = all(
            all(candidate_flags[baseline_name].values())
            for baseline_name in baseline
        )
        flags[candidate_id] = candidate_flags
    decision = (
        "NO_MOUNT_ACCEPTED_DIAGNOSTIC_RECONSTRUCTION"
        if not any(item["accepted"] for item in flags.values())
        else "ACCEPTED_FOR_CONTRACT_REGENERATION_ONLY"
    )
    summary = {
        "schema_version": "kcg_d38999_diag_posthoc_comparison_v1",
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "scope": "DIAGNOSTIC_RECONSTRUCTION_ONLY",
        "decision": decision,
        "baselines": baseline,
        "variants": variants,
        "acceptance_flags": flags,
        "control_authorized": False,
        "mount_formally_accepted": False,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-sidecar", action="store_true")
    parser.add_argument("--build-truth-free-sidecar", action="store_true")
    parser.add_argument("--formal-only", action="store_true")
    parser.add_argument("--posthoc-compare", action="store_true")
    parser.add_argument(
        "--base",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_diag_mount_search_v1/seed000/postgrasp_diag_mount_search"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_diag_mount_search_v1/seed000/nominal_physics_report.json"
        ),
    )
    parser.add_argument(
        "--controller-steps",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "phase1_diag_mount_search_v1/seed000/controller_steps.jsonl"
        ),
    )
    parser.add_argument(
        "--pick-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    parser.add_argument(
        "--sidecar",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/diag_capture_contract_reconstructed.json"
        ),
    )
    parser.add_argument(
        "--formal-output",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/offline_diag_mount_formal"
        ),
    )
    parser.add_argument(
        "--comparison-output",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
            / "deepseek/offline_diag_mount_comparison.json"
        ),
    )
    args = parser.parse_args()
    if args.build_truth_free_sidecar:
        build_truth_free_capture_sidecar(
            base_path=args.base,
            output_path=args.sidecar,
        )
        print("TRUTH_FREE_DIAGNOSTIC_SIDECAR_BUILT")
        return 0
    if args.build_sidecar:
        build_diagnostic_reconstruction_sidecar(
            base_path=args.base,
            report_path=args.report,
            controller_steps_path=args.controller_steps,
            pick_config_path=args.pick_config,
            output_path=args.sidecar,
        )
        print("SIDECAR_BUILT")
        return 0
    if args.formal_only:
        result = run_formal_only(
            contract_path=args.sidecar,
            raw_root=args.base,
            output_path=args.formal_output,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    if args.posthoc_compare:
        result = run_posthoc_comparison(
            report_path=args.report,
            formal_output_path=args.formal_output,
            output_path=args.comparison_output,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    parser.error("choose --build-sidecar, --formal-only, or --posthoc-compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
