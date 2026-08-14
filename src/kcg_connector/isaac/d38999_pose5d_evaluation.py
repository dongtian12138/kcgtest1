#!/usr/bin/env python3

"""Thirty-scene semantic RGB-D Pose5D/C2 evaluation in Isaac Sim.

Endpoint transforms are randomized only before physics starts.  Semantic masks,
registered depth, and a calibrated nominal +Z prior are the estimator inputs.
USD truth poses are read after estimation solely to calculate withheld errors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback

import numpy as np


def _arguments(repository: Path):
    parser = argparse.ArgumentParser(description="D38999 Pose5D/C2 RGB-D evaluation")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--show-pose5d", action="store_true")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=38999)
    parser.add_argument(
        "--rgbd-config",
        default=str(repository / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"),
    )
    parser.add_argument(
        "--pose5d-config",
        default=str(repository / "src/kcg_connector/config/d38999_pose5d_v1.yaml"),
    )
    parser.add_argument(
        "--tolerance-report",
        default=str(repository / "artifacts/kcg_connector/d38999_insert_proxy_v2/tolerance_sweep.json"),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/kcg_connector/d38999_visual_ft_e2e_v1/pose5d_evaluation",
    )
    result = parser.parse_args()
    if not result.run:
        parser.error("Pose5D evaluation requires explicit --run")
    if not result.show_pose5d:
        parser.error("Pose5D evaluation requires --show-pose5d")
    if result.keep_open and not result.gui:
        parser.error("--keep-open requires --gui")
    if result.episodes < 30:
        parser.error("Pose5D evaluation requires at least 30 episodes")
    return result


def _output_path(repository: Path, value: str) -> Path:
    path = Path(value).expanduser()
    result = path.resolve() if path.is_absolute() else (repository / path).resolve()
    if repository not in result.parents:
        raise ValueError("output directory must remain below the repository")
    return result


def _scene(tabletop, loose_xy, fixed_xy, loose_z_offset):
    fixed_delta = (
        fixed_xy[0] - tabletop.fixed_endpoint.receptacle_origin_m[0],
        fixed_xy[1] - tabletop.fixed_endpoint.receptacle_origin_m[1],
    )
    fixture = tabletop.fixed_endpoint.fixture_center_m
    fixed = replace(
        tabletop.fixed_endpoint,
        fixture_center_m=(fixture[0] + fixed_delta[0], fixture[1] + fixed_delta[1], fixture[2]),
        receptacle_origin_m=(fixed_xy[0], fixed_xy[1], tabletop.fixed_endpoint.receptacle_origin_m[2]),
    )
    loose = replace(
        tabletop.loose_endpoint,
        initial_origin_m=(loose_xy[0], loose_xy[1], tabletop.loose_endpoint.initial_origin_m[2] + loose_z_offset),
    )
    return replace(tabletop, fixed_endpoint=fixed, loose_endpoint=loose)


def _author_orientation(stage, path, rpy, UsdGeom):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"missing endpoint prim {path}")
    xformable = UsdGeom.Xformable(prim)
    operations = xformable.GetOrderedXformOps()
    if len(operations) != 1 or operations[0].GetOpType() != UsdGeom.XformOp.TypeTranslate:
        raise RuntimeError(f"endpoint transform is not translation-only: {path}")
    xformable.AddRotateXYZOp().Set(tuple(math.degrees(value) for value in rpy))


def _axis_from_wxyz(quaternion):
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)))


def _axis_error(a, b):
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def _summary(values):
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(np.max(data)),
    }


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    output_dir = _output_path(repository, arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": not arguments.gui,
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
    })
    completed = False
    report = {
        "schema_version": "kcg_d38999_pose5d_evaluation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "required_episode_count": arguments.episodes,
        "random_seed": arguments.seed,
        "estimator_truth_inputs": [],
        "truth_scope": "post_hoc_evaluation_only",
        "episodes": [],
        "passed": False,
    }
    try:
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import add_labels, get_labels
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from kcg_connector.d38999_pose5d import load_pose5d_config
        from kcg_connector.d38999_tabletop_scene import author_d38999_tabletop_scene, load_d38999_tabletop_scene, verify_d38999_tabletop_asset
        from kcg_connector.isaac_d38999_rgbd_runtime import capture_d38999_rgbd_runtime
        from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap

        rgbd = load_rgbd_bootstrap(arguments.rgbd_config)
        pose5d_config = load_pose5d_config(arguments.pose5d_config)
        tabletop = load_d38999_tabletop_scene(repository / rgbd.tabletop_config)
        asset_path = verify_d38999_tabletop_asset(tabletop, repository)
        tolerance = json.loads(Path(arguments.tolerance_report).read_text(encoding="utf-8"))
        boundaries = tolerance["measured_boundaries"]
        lateral_gate = min(boundaries["x_offset_m"]["authorization_gate_abs"], boundaries["y_offset_m"]["authorization_gate_abs"])
        axis_gate = min(boundaries["tilt_x_rad"]["authorization_gate_abs"], boundaries["tilt_y_rad"]["authorization_gate_abs"])
        gates = {"lateral_position_m": lateral_gate, "axis_angle_rad": axis_gate}
        if not lateral_gate > 0.0 or not axis_gate > 0.0:
            raise RuntimeError("V2 measured authorization gates are unavailable")
        report["authorization_gates"] = gates
        bindings = {
            "Camera": Camera, "Gf": Gf, "Image": Image, "Usd": Usd,
            "UsdGeom": UsdGeom, "UsdLux": UsdLux, "add_labels": add_labels,
            "get_labels": get_labels, "rep": rep,
        }
        rng = np.random.default_rng(arguments.seed)
        for index in range(arguments.episodes):
            episode = {"episode_index": index, "capture_completed": False}
            world = None
            try:
                loose_xy = (0.520 + rng.uniform(-0.025, 0.025), -0.210 + rng.uniform(-0.025, 0.025))
                fixed_xy = (0.550 + rng.uniform(-0.012, 0.012), 0.185 + rng.uniform(-0.012, 0.012))
                loose_rpy = tuple(rng.uniform(-math.radians(5.0), math.radians(5.0), 2)) + (float(rng.uniform(-math.pi, math.pi)),)
                fixed_rpy = tuple(rng.uniform(-math.radians(5.0), math.radians(5.0), 2)) + (float(rng.uniform(-math.pi, math.pi)),)
                scene = _scene(tabletop, loose_xy, fixed_xy, 0.004)
                World.clear_instance()
                create_new_stage()
                simulation_app.update()
                world = World(stage_units_in_meters=1.0, physics_dt=1.0 / scene.physics.rate_hz, rendering_dt=1.0 / 60.0, backend="numpy", device="cpu")
                stage = get_current_stage()
                author_d38999_tabletop_scene(
                    stage, scene, asset_path,
                    add_reference_to_stage=add_reference_to_stage,
                    Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
                    UsdShade=UsdShade, physics_utils=physicsUtils,
                )
                _author_orientation(stage, scene.asset.loose_plug_prim_path, loose_rpy, UsdGeom)
                _author_orientation(stage, scene.asset.fixed_receptacle_prim_path, fixed_rpy, UsdGeom)
                loose_prim = stage.GetPrimAtPath(scene.asset.loose_plug_prim_path)
                fixed_prim = stage.GetPrimAtPath(scene.asset.fixed_receptacle_prim_path)
                body = world.scene.add(SingleRigidPrim(prim_path=scene.asset.body_prim_path, name=f"pose5d_body_{index:03d}"))
                world.reset()
                world.get_physics_context().set_gravity(0.0)
                for _ in range(2):
                    world.step(render=True)
                capture_id = f"pose5d-{arguments.seed}-{index:03d}"
                capture = capture_d38999_rgbd_runtime(
                    bindings=bindings, simulation_app=simulation_app, world=world,
                    stage=stage, tabletop=scene, rgbd=rgbd, loose_prim=loose_prim,
                    fixed_prim=fixed_prim, body=body,
                    output_dir=output_dir / f"episode_{index:03d}",
                    pose5d_config=pose5d_config, pose5d_capture_id=capture_id,
                    pose5d_axis_priors={"loose_plug": (0.0, 0.0, 1.0), "fixed_receptacle": (0.0, 0.0, 1.0)},
                    pose5d_authorization_gates=gates,
                )
                if "pose5d" not in capture.metrics:
                    raise RuntimeError(capture.metrics.get("error", "Pose5D output missing"))
                pose = capture.metrics["pose5d"]
                truth = {
                    "loose_plug": (np.asarray(capture.loose_position_world_m), _axis_from_wxyz(capture.loose_orientation_wxyz)),
                    "fixed_receptacle": (np.asarray(capture.fixed_position_world_m), _axis_from_wxyz(capture.fixed_orientation_wxyz)),
                }
                endpoint_eval = {}
                for role in ("loose_plug", "fixed_receptacle"):
                    estimate = pose[role]
                    estimated_center = np.asarray(estimate["xyz_m"])
                    estimated_axis = np.asarray(estimate["axis_vector"])
                    true_center, true_axis = truth[role]
                    error = estimated_center - true_center
                    axial_error = float(np.dot(error, true_axis))
                    lateral_error = error - axial_error * true_axis
                    endpoint_eval[role] = {
                        "xyz_error_m": float(np.linalg.norm(error)),
                        "lateral_position_error_m": float(np.linalg.norm(lateral_error)),
                        "axial_position_error_m": abs(axial_error),
                        "axis_error_rad": _axis_error(estimated_axis, true_axis),
                    }
                estimated_delta = np.asarray(pose["fixed_receptacle"]["xyz_m"]) - np.asarray(pose["loose_plug"]["xyz_m"])
                true_delta = truth["fixed_receptacle"][0] - truth["loose_plug"][0]
                relative_error = estimated_delta - true_delta
                fixed_axis = truth["fixed_receptacle"][1]
                relative_lateral = relative_error - float(np.dot(relative_error, fixed_axis)) * fixed_axis
                authorized = bool(pose["loose_plug"]["control_authorized"] and pose["fixed_receptacle"]["control_authorized"])
                within_gate = bool(
                    np.linalg.norm(relative_lateral) <= lateral_gate
                    and endpoint_eval["loose_plug"]["axis_error_rad"] <= axis_gate
                    and endpoint_eval["fixed_receptacle"]["axis_error_rad"] <= axis_gate
                )
                episode.update({
                    "capture_completed": True,
                    "capture_id": capture_id,
                    "randomization": {"loose_xyz_rpy": [*loose_xy, scene.loose_endpoint.initial_origin_m[2], *loose_rpy], "fixed_xyz_rpy": [*fixed_xy, scene.fixed_endpoint.receptacle_origin_m[2], *fixed_rpy]},
                    "pose5d": pose,
                    "post_hoc_truth_evaluation": endpoint_eval,
                    "relative_lateral_error_m": float(np.linalg.norm(relative_lateral)),
                    "estimator_control_authorized": authorized,
                    "post_hoc_within_measured_gate": within_gate,
                    "false_authorization": bool(authorized and not within_gate),
                    "truth_used_by_estimator": False,
                })
            except BaseException as exception:
                episode["error"] = f"{type(exception).__name__}: {exception}"
                episode["traceback"] = traceback.format_exc()
            finally:
                if world is not None:
                    try:
                        world.stop()
                        simulation_app.update()
                    except BaseException as exception:
                        episode["world_stop_error"] = f"{type(exception).__name__}: {exception}"
                World.clear_instance()
            report["episodes"].append(episode)
            print(json.dumps(_finite_json(episode), allow_nan=False, sort_keys=True), flush=True)

        valid = [item for item in report["episodes"] if item.get("capture_completed")]
        report["completed_episode_count"] = len(valid)
        report["rejected_count"] = sum(not item["estimator_control_authorized"] for item in valid)
        report["false_authorization_count"] = sum(item["false_authorization"] for item in valid)
        report["rejection_rate"] = report["rejected_count"] / max(1, len(valid))
        report["false_authorization_rate"] = report["false_authorization_count"] / max(1, len(valid))
        for role in ("loose_plug", "fixed_receptacle"):
            report.setdefault("error_statistics", {})[role] = {
                "xyz_error_m": _summary([item["post_hoc_truth_evaluation"][role]["xyz_error_m"] for item in valid]),
                "axis_error_rad": _summary([item["post_hoc_truth_evaluation"][role]["axis_error_rad"] for item in valid]),
            }
        report["relative_lateral_error_m"] = _summary([item["relative_lateral_error_m"] for item in valid])
        completed = len(valid) == arguments.episodes
        report["passed"] = completed
        flat_rows = []
        for item in valid:
            row = {
                "episode_index": item["episode_index"],
                "authorized": item["estimator_control_authorized"],
                "false_authorization": item["false_authorization"],
                "relative_lateral_error_m": item["relative_lateral_error_m"],
            }
            for role in ("loose_plug", "fixed_receptacle"):
                for key, value in item["post_hoc_truth_evaluation"][role].items():
                    row[f"{role}_{key}"] = value
            flat_rows.append(row)
        with (output_dir / "pose5d_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
        (output_dir / "pose5d_evaluation.json").write_text(json.dumps(_finite_json(report), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("passed", "completed_episode_count", "rejected_count", "false_authorization_count", "error_statistics")}, sort_keys=True), flush=True)
        print("ISAAC D38999 POSE5D EVALUATION " + ("COMPLETED" if completed else "FAILED"), flush=True)
        if arguments.keep_open:
            while simulation_app.is_running():
                simulation_app.update()
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        report["traceback"] = traceback.format_exc()
        (output_dir / "pose5d_evaluation_failed.json").write_text(json.dumps(_finite_json(report), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
