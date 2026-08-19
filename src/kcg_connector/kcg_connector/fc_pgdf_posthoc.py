"""CPU-only posthoc evaluator for the FC-PGDF-01a raw-capture smoke.

The formal archive contains only RGB, depth, camera calibration, robot FK
and sensor state.  This evaluator is allowed to read the snapshot gate truth
and the separate posthoc truth sidecar; it is explicitly
``POSTHOC_TRUTH_ONLY`` and never feeds an estimator or controller.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import (
    PLUG_MATING,
    CameraModel,
    fixed_camera_model,
    project,
    proxy_cad_points,
)
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
)
from kcg_connector.postgrasp_snapshot_gate import load_snapshot_gate_document

SCHEMA_VERSION = "kcg_d38999_fc_pgdf_01a_posthoc_v1"
MINIMUM_PLUG_FACE_SHORT_AXIS_PX = 80.0
MINIMUM_DIRECTION_DIFFERENCE_DEG = 15.0
MAXIMUM_POSTHOC_SLIP_TRANSLATION_M = 0.005
MAXIMUM_POSTHOC_SLIP_ROTATION_DEG = 5.0
THRESHOLD_LABEL = "SIM_TUNING_ONLY_CANDIDATE"


def _matrix_from_wxyz(orientation_wxyz: Sequence[float]) -> np.ndarray:
    value = np.asarray(orientation_wxyz, dtype=np.float64)
    value = value / np.linalg.norm(value)
    return Rotation.from_quat(
        [value[1], value[2], value[3], value[0]]
    ).as_matrix()


def _load_json(path: Path, role: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("role") != role:
        raise ValueError(f"{path} role is not {role}")
    return document


def _camera_from_archive(camera_path: Path) -> CameraModel:
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    return fixed_camera_model(
        eye=tuple(float(value) for value in camera["eye_m"]),
        target=tuple(float(value) for value in camera["target_m"]),
        resolution=(640, 480),
        focal_length_mm=24.0,
        horizontal_aperture_mm=20.955,
    )


def _project_face_pixels(
    camera: CameraModel,
    plug_cad,
    plug_transform: np.ndarray,
    depth_image: np.ndarray,
):
    points = plug_cad.xyz[plug_cad.label == PLUG_MATING]
    world_points = (
        plug_transform[:3, :3] @ points.T
    ).T + plug_transform[:3, 3]
    uv, predicted_depth = project(camera, world_points)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height)
        & (predicted_depth > 0.0)
        & np.isfinite(predicted_depth)
    )
    u = uv[inside]
    predicted = predicted_depth[inside]
    if len(u) < 2:
        return {
            "projected_inside_pixels": int(np.sum(inside)),
            "bbox_width_px": 0.0,
            "bbox_height_px": 0.0,
            "short_axis_px": 0.0,
            "face_visible_fraction": 0.0,
            "depth_valid_fraction_in_bbox": 0.0,
            "visible_face_pixels": 0,
        }
    ui = np.clip(u[:, 0].astype(np.int64), 0, camera.width - 1)
    vi = np.clip(u[:, 1].astype(np.int64), 0, camera.height - 1)
    observed_depth = depth_image[vi, ui]
    observed_valid = np.isfinite(observed_depth) & (observed_depth > 0.0)
    visible = observed_valid & (predicted <= observed_depth + 0.015)
    bbox_width = float(np.max(u[:, 0]) - np.min(u[:, 0]))
    bbox_height = float(np.max(u[:, 1]) - np.min(u[:, 1]))
    x0 = int(np.floor(np.min(u[:, 0])))
    x1 = int(np.ceil(np.max(u[:, 0])))
    y0 = int(np.floor(np.min(u[:, 1])))
    y1 = int(np.ceil(np.max(u[:, 1])))
    crop = depth_image[
        max(0, y0):min(camera.height, y1 + 1),
        max(0, x0):min(camera.width, x1 + 1),
    ]
    crop_valid_fraction = (
        float(np.mean(np.isfinite(crop) & (crop > 0.0)))
        if crop.size
        else 0.0
    )
    return {
        "projected_inside_pixels": int(len(u)),
        "bbox_width_px": bbox_width,
        "bbox_height_px": bbox_height,
        "short_axis_px": min(bbox_width, bbox_height),
        "face_visible_fraction": float(np.mean(visible)),
        "depth_valid_fraction_in_bbox": crop_valid_fraction,
        "visible_face_pixels": int(np.sum(visible)),
    }


def evaluate_fc_pgdf_01a_posthoc(
    *,
    snapshot_path: Path | str,
    output_root: Path | str,
    pick_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate one FC-PGDF-01a smoke output and write the posthoc report."""

    repository = Path(__file__).resolve().parents[3]
    output_root = Path(output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    snapshot = load_snapshot_gate_document(snapshot_path)
    if pick_config_path is None:
        pick_config_path = (
            repository
            / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        )
    pick = load_d38999_tabletop_pick_config(pick_config_path)
    plug_cad, _ = proxy_cad_points()

    formal_manifest = _load_json(
        output_root / "formal_manifest.json", "formal_raw_observation"
    )
    sidecar = _load_json(
        output_root / "posthoc_truth_sidecar.json", "posthoc_truth_sidecar"
    )
    if formal_manifest.get("object_truth_present") is not False:
        raise ValueError("formal archive must declare object truth absent")
    if formal_manifest.get("contact_report_present") is not False:
        raise ValueError("formal archive must declare contact report absent")
    if formal_manifest.get("control_authorized") is not False:
        raise ValueError("formal archive control_authorized must be false")
    if sidecar.get("formal_estimator_input") is not False:
        raise ValueError("posthoc sidecar must not be formal estimator input")
    if sidecar.get("control_authorized") is not False:
        raise ValueError("posthoc sidecar must not authorize control")

    tcp_from_handbase = np.eye(4, dtype=np.float64)
    tcp_from_handbase[2, 3] = -float(
        pick.geometry_candidate.handbase_to_tcp_m
    )

    snapshot_arm_q = np.asarray(
        snapshot["robot_state"]["q_rad"][:7], dtype=np.float64
    )
    snapshot_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(snapshot_arm_q))
    )
    snapshot_hand = snapshot_tcp @ tcp_from_handbase
    snapshot_plug = np.eye(4, dtype=np.float64)
    snapshot_plug[:3, :3] = _matrix_from_wxyz(
        snapshot["plug_root_state"]["orientation_wxyz"]
    )
    snapshot_plug[:3, 3] = np.asarray(
        snapshot["plug_root_state"]["position_m"], dtype=np.float64
    )
    t_hand_plug_snapshot = np.linalg.inv(snapshot_hand) @ snapshot_plug

    view_reports = []
    directions = []
    for view in formal_manifest["views"]:
        view_id = view["view_id"]
        view_dir = output_root / "formal_views" / view_id
        fk = json.loads((view_dir / "fk.json").read_text(encoding="utf-8"))
        camera = _camera_from_archive(view_dir / "camera.json")
        depth = np.load(view_dir / "depth_m.npy")
        arm_q = np.asarray(fk["arm_q_actual_rad"], dtype=np.float64)
        view_tcp = np.asarray(
            iiwa14_grasp_tcp_transform(tuple(arm_q))
        )
        view_hand = view_tcp @ tcp_from_handbase
        view_plug = view_hand @ t_hand_plug_snapshot
        face = _project_face_pixels(camera, plug_cad, view_plug, depth)
        plug_center = view_plug[:3, 3]
        camera_to_plug = (
            np.asarray(camera.position_world, dtype=np.float64) - plug_center
        )
        camera_distance = float(np.linalg.norm(camera_to_plug))
        direction = camera_to_plug / camera_distance
        directions.append(direction)
        view_reports.append(
            {
                "view_id": view_id,
                "posthoc_plug_center_m": plug_center.tolist(),
                "camera_distance_m": camera_distance,
                "camera_to_plug_direction": direction.tolist(),
                **face,
            }
        )

    direction_difference_deg = float(
        math.degrees(
            math.acos(
                max(-1.0, min(1.0, float(np.dot(directions[0], directions[1]))))
            )
        )
    )

    final_plug = np.eye(4, dtype=np.float64)
    final_plug[:3, :3] = _matrix_from_wxyz(
        sidecar["final_plug_orientation_wxyz"]
    )
    final_plug[:3, 3] = np.asarray(
        sidecar["final_plug_position_m"], dtype=np.float64
    )
    final_view = view_reports[-1]
    final_fk = json.loads(
        (
            output_root
            / "formal_views"
            / final_view["view_id"]
            / "fk.json"
        ).read_text(encoding="utf-8")
    )
    final_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(
            tuple(float(value) for value in final_fk["arm_q_actual_rad"])
        )
    )
    final_hand = final_tcp @ tcp_from_handbase
    t_hand_plug_final = np.linalg.inv(final_hand) @ final_plug
    relative_hand_plug = (
        np.linalg.inv(t_hand_plug_snapshot) @ t_hand_plug_final
    )
    slip_translation_m = float(
        np.linalg.norm(relative_hand_plug[:3, 3])
    )
    slip_rotation_deg = float(
        math.degrees(
            np.linalg.norm(
                Rotation.from_matrix(
                    relative_hand_plug[:3, :3]
                ).as_rotvec()
            )
        )
    )

    max_short_axis_px = max(
        float(view["short_axis_px"]) for view in view_reports
    )
    pixel_gate = max_short_axis_px >= MINIMUM_PLUG_FACE_SHORT_AXIS_PX
    direction_gate = (
        direction_difference_deg >= MINIMUM_DIRECTION_DIFFERENCE_DEG
    )
    slip_gate = (
        slip_translation_m <= MAXIMUM_POSTHOC_SLIP_TRANSLATION_M
        and slip_rotation_deg <= MAXIMUM_POSTHOC_SLIP_ROTATION_DEG
    )
    if not pixel_gate:
        status = "CURRENT_FIXED_CAMERA_PIXEL_INFEASIBLE"
        conclusion = (
            "两个展示姿态的 Plug 正脸短轴均明显小于 80 px；"
            "按约定不降门，直接判定现有固定相机像素不可行。"
        )
    elif direction_gate and slip_gate:
        status = "PASS"
        conclusion = (
            "两个展示姿态达到候选像素、方向差和抓持稳定要求。"
        )
    elif direction_gate or slip_gate:
        status = "MARGINAL"
        conclusion = (
            "两个展示姿态只满足部分候选门，需要调整后复验。"
        )
    else:
        status = "FAIL"
        conclusion = "两个展示姿态未满足候选可行性门。"

    report = {
        "schema_version": SCHEMA_VERSION,
        "role": "posthoc_evaluation",
        "truth_scope": "POSTHOC_TRUTH_ONLY",
        "formal_estimator_input": False,
        "control_authorized": False,
        "status": status,
        "conclusion_zh": conclusion,
        "threshold_label": THRESHOLD_LABEL,
        "pixel_gate": pixel_gate,
        "maximum_plug_face_short_axis_px": max_short_axis_px,
        "minimum_plug_face_short_axis_px_candidate": (
            MINIMUM_PLUG_FACE_SHORT_AXIS_PX
        ),
        "direction_difference_deg": direction_difference_deg,
        "minimum_direction_difference_deg_candidate": (
            MINIMUM_DIRECTION_DIFFERENCE_DEG
        ),
        "direction_gate": direction_gate,
        "posthoc_slip_translation_m": slip_translation_m,
        "posthoc_slip_rotation_deg": slip_rotation_deg,
        "slip_gate": slip_gate,
        "object_pose_writes_after_restore": sidecar[
            "object_pose_writes_after_restore"
        ],
        "views": view_reports,
    }
    (output_root / "fc_pgdf_posthoc_report.json").write_text(
        json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pick-config",
        default=(
            Path(__file__).resolve().parents[3]
            / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
        ),
    )
    arguments = parser.parse_args()
    report = evaluate_fc_pgdf_01a_posthoc(
        snapshot_path=arguments.snapshot,
        output_root=arguments.output_dir,
        pick_config_path=arguments.pick_config,
    )
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
