#!/usr/bin/env python3

"""Capture standalone evidence through the reusable in-World RGB-D helper.

This is a camera-pipeline bootstrap, not a learned detector.  The exact same
runtime capture helper is used by the opt-in end-to-end pose preflight, which
prevents the standalone proof and integrated episode from drifting into two
different semantic or estimator implementations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


MINIMUM_ENDPOINT_CENTER_MARGIN_PX = 16


def _endpoint_projection_records(endpoint_uv, resolution):
    """Compatibility wrapper around the pure reusable runtime helper."""
    from kcg_connector.isaac_d38999_rgbd_runtime import (
        endpoint_projection_records,
    )

    return endpoint_projection_records(endpoint_uv, resolution)


def _validate_real_endpoint_semantic_ids(endpoint_ids, observed_ids):
    """Compatibility wrapper around the pure reusable runtime helper."""
    from kcg_connector.isaac_d38999_rgbd_runtime import (
        validate_real_endpoint_semantic_ids,
    )

    return validate_real_endpoint_semantic_ids(endpoint_ids, observed_ids)


def _arguments(repository):
    parser = argparse.ArgumentParser(
        description="Capture the D38999 tabletop RGB-D bootstrap evidence"
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"
        ),
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--capture-count",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "capture once normally, or twice in the same World to verify "
            "persistent camera/light reuse"
        ),
    )
    arguments = parser.parse_args()
    if arguments.keep_open and not arguments.gui:
        parser.error("--keep-open requires --gui")
    return arguments


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    metrics = {
        "camera_observation_present": False,
        "detector_kind": "isaac_renderer_semantic_annotation_bootstrap",
        "foundation_pose_present": False,
        "full_keyed_6d_vision_pose_claimed": False,
        "gui": arguments.gui,
        "learned_detector_present": False,
        "object_pose_writes_after_start": 0,
        "passed": False,
        "real_camera_present": False,
        "scene": "kcg_d38999_rgbd_bootstrap_v1",
    }
    try:
        from PIL import Image

        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import (
            add_labels,
            get_labels,
        )
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from kcg_connector.connector_pose import (
            load_connector_pose_contract,
            pair_connector_pose_observations,
        )
        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
            load_d38999_tabletop_scene,
            verify_d38999_tabletop_asset,
        )
        from kcg_connector.isaac_d38999_rgbd_runtime import (
            capture_d38999_rgbd_runtime,
        )
        from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap
        from kcg_connector.sim_pose_provider import (
            make_sim_ground_truth_observation,
        )

        rgbd = load_rgbd_bootstrap(arguments.config)
        tabletop_path = repository / rgbd.tabletop_config
        pose_contract_path = repository / rgbd.pose_contract_config
        tabletop = load_d38999_tabletop_scene(tabletop_path)
        asset_path = verify_d38999_tabletop_asset(tabletop, repository)
        output_dir = repository / rgbd.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / tabletop.physics.rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        metrics["authored_scene"] = author_d38999_tabletop_scene(
            stage,
            tabletop,
            asset_path,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        loose_prim = stage.GetPrimAtPath(
            tabletop.asset.loose_plug_prim_path
        )
        fixed_prim = stage.GetPrimAtPath(
            tabletop.asset.fixed_receptacle_prim_path
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=tabletop.asset.body_prim_path,
                name="rgbd_loose_body",
            )
        )
        world.reset()
        world.get_physics_context().set_gravity(
            tabletop.physics.gravity_m_s2
        )
        for _ in range(tabletop.physics.settle_steps):
            world.step(render=True)

        runtime_bindings = {
            "Camera": Camera,
            "Gf": Gf,
            "Image": Image,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "UsdLux": UsdLux,
            "add_labels": add_labels,
            "get_labels": get_labels,
            "rep": rep,
        }
        capture_history = []
        capture = None
        # The optional second pass stays in this exact World and episode.  It
        # is a narrow runtime-lifecycle check: no reset, clear or physics step
        # is inserted between calls, and repeated artifacts use a subfolder.
        for capture_index in range(arguments.capture_count):
            capture_output_dir = (
                output_dir
                if capture_index == 0
                else output_dir / f"repeat_{capture_index + 1:02d}"
            )
            capture = capture_d38999_rgbd_runtime(
                bindings=runtime_bindings,
                simulation_app=simulation_app,
                world=world,
                stage=stage,
                tabletop=tabletop,
                rgbd=rgbd,
                loose_prim=loose_prim,
                fixed_prim=fixed_prim,
                body=body,
                output_dir=capture_output_dir,
            )
            capture_history.append(
                {
                    "capture_index": capture_index + 1,
                    "passed": capture.passed,
                    "resource_cleanup": capture.metrics.get(
                        "resource_cleanup"
                    ),
                    "stage_prim_lifecycle": capture.metrics.get(
                        "stage_prim_lifecycle"
                    ),
                    "timeline_state": capture.metrics.get(
                        "timeline_state"
                    ),
                }
            )
            if capture.passed is not True:
                break

        if capture is None:  # Defensive: argparse only accepts 1 or 2.
            raise RuntimeError("RGB-D capture loop did not execute")
        metrics.update(capture.metrics)
        metrics["capture_count_completed"] = len(capture_history)
        metrics["capture_count_requested"] = arguments.capture_count
        metrics["capture_history"] = capture_history
        metrics["scene"] = "kcg_d38999_rgbd_bootstrap_v1"
        metrics["gui"] = arguments.gui
        if capture.passed is not True:
            raise RuntimeError(
                "reusable D38999 RGB-D runtime capture failed: "
                f"{capture.metrics.get('error', 'acceptance gate')}"
            )

        pose_contract = load_connector_pose_contract(pose_contract_path)
        timestamp_s = (
            tabletop.physics.settle_duration_s
            + rgbd.camera.warmup_frames / tabletop.physics.rate_hz
        )
        loose_pose = make_sim_ground_truth_observation(
            pose_contract,
            model_id="d38999_26kj61sn_proxy_v1",
            role="loose_plug",
            timestamp_s=timestamp_s,
            now_s=timestamp_s,
            frame_id="world",
            position_xyz_m=capture.loose_position_world_m,
            quaternion_wxyz=capture.loose_orientation_wxyz,
        )
        fixed_pose = make_sim_ground_truth_observation(
            pose_contract,
            model_id="d38999_20kj61pn_proxy_v1",
            role="fixed_receptacle",
            timestamp_s=timestamp_s,
            now_s=timestamp_s,
            frame_id="world",
            position_xyz_m=capture.fixed_position_world_m,
            quaternion_wxyz=capture.fixed_orientation_wxyz,
        )
        pose_pair = pair_connector_pose_observations(
            loose_pose, fixed_pose, pose_contract, now_s=timestamp_s
        )
        metrics["pose_contract"] = {
            "full_rgbd_pose_accepted": False,
            "pair_valid": bool(
                pose_pair.loose_plug.model_id
                == "d38999_26kj61sn_proxy_v1"
                and pose_pair.fixed_receptacle.model_id
                == "d38999_20kj61pn_proxy_v1"
            ),
            "rejection_reason": (
                "key_yaw_unobservable_and_renderer_annotation_is_not_a_"
                "learned_detector"
            ),
            "source": pose_pair.loose_plug.source.value,
        }
        passed = bool(
            capture.passed
            and metrics["pose_contract"]["pair_valid"]
            and metrics["pose_contract"]["full_rgbd_pose_accepted"] is False
            and metrics["object_pose_writes_after_start"] == 0
        )
        metrics["passed"] = passed
        report_path = output_dir / rgbd.output.report_filename
        report_path.write_text(
            json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, allow_nan=False, sort_keys=True), flush=True)
        print(
            "ISAAC D38999 RGBD BOOTSTRAP V1 "
            + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        metrics.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
            }
        )
        traceback.print_exc()
        print(json.dumps(metrics, allow_nan=False, sort_keys=True), flush=True)
        print("ISAAC D38999 RGBD BOOTSTRAP V1 FAILED", flush=True)
    finally:
        if arguments.keep_open and arguments.gui:
            print(
                "ISAAC D38999 RGBD BOOTSTRAP V1 GUI REMAINS OPEN; "
                "close the window to exit",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
