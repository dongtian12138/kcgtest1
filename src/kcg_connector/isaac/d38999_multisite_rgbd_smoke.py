#!/usr/bin/env python3

"""Opt-in five-anchor RGB-D sweep for the D38999 tabletop proxy.

One ``SimulationApp`` is reused to avoid five expensive Kit startups, but each
anchor receives a fresh USD stage and ``World``.  Endpoint XY/yaw and the
matching fixed fixture position are authored before ``world.reset()``.  Once
physics starts, this script performs no endpoint pose writes.  It validates
only semantic RGB-D visibility and ray-plane XY; keyed yaw, full 6D pose, and
robot-control authorization remain explicitly false.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import traceback


REPORT_SCHEMA_VERSION = "kcg_d38999_multisite_rgbd_report_v1"
STRICT_MAXIMUM_XY_ERROR_M = 0.010


def _arguments(repository: Path):
    parser = argparse.ArgumentParser(
        description=(
            "Run the disabled-by-default D38999 five-anchor RGB-D sweep"
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_multisite_vision6d_v1.yaml"
        ),
        help="strict multi-position vision preparation contract",
    )
    parser.add_argument(
        "--rgbd-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_rgbd_bootstrap_v1.yaml"
        ),
        help="existing fixed-global-camera RGB-D contract",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/kcg_connector/d38999_multisite_rgbd_v1",
        help="repository-relative trial artifacts and report directory",
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def _repository_output_path(repository: Path, value: str) -> Path:
    """Resolve one repository-local output without allowing path escape."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("--output-dir must be repository-relative")
    result = (repository / relative).resolve()
    if repository not in result.parents or result == repository:
        raise ValueError("--output-dir must be below the repository root")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_for_anchor(tabletop, anchor):
    """Return an immutable scene copy with one anchor's pre-physics XY."""
    loose_xy = tuple(float(value) for value in anchor["loose_xy_m"])
    fixed_xy = tuple(float(value) for value in anchor["fixed_xy_m"])
    fixed_delta = (
        fixed_xy[0] - tabletop.fixed_endpoint.receptacle_origin_m[0],
        fixed_xy[1] - tabletop.fixed_endpoint.receptacle_origin_m[1],
    )
    fixture_center = tabletop.fixed_endpoint.fixture_center_m
    fixed_endpoint = replace(
        tabletop.fixed_endpoint,
        fixture_center_m=(
            fixture_center[0] + fixed_delta[0],
            fixture_center[1] + fixed_delta[1],
            fixture_center[2],
        ),
        receptacle_origin_m=(
            fixed_xy[0],
            fixed_xy[1],
            tabletop.fixed_endpoint.receptacle_origin_m[2],
        ),
    )
    loose_endpoint = replace(
        tabletop.loose_endpoint,
        initial_origin_m=(
            loose_xy[0],
            loose_xy[1],
            tabletop.loose_endpoint.initial_origin_m[2],
        ),
    )
    return replace(
        tabletop,
        fixed_endpoint=fixed_endpoint,
        loose_endpoint=loose_endpoint,
    )


def _author_endpoint_yaws_before_physics(
    *, stage, tabletop, loose_yaw_rad, fixed_yaw_rad, UsdGeom
):
    """Author exactly one yaw op per endpoint before physics starts.

    ``author_d38999_tabletop_scene`` has already authored a single translation
    operation for each endpoint root.  Appending rotate-Z preserves the root
    XY translation while rotating all child visuals and colliders together.
    This helper is never called after ``world.reset()``.
    """
    records = {}
    for role, path, yaw_rad in (
        (
            "loose_plug",
            tabletop.asset.loose_plug_prim_path,
            loose_yaw_rad,
        ),
        (
            "fixed_receptacle",
            tabletop.asset.fixed_receptacle_prim_path,
            fixed_yaw_rad,
        ),
    ):
        value = float(yaw_rad)
        if not math.isfinite(value) or not -math.pi <= value <= math.pi:
            raise ValueError(f"{role} yaw must be finite within [-pi, pi]")
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"endpoint prim is missing: {path}")
        xformable = UsdGeom.Xformable(prim)
        operations = xformable.GetOrderedXformOps()
        if (
            len(operations) != 1
            or operations[0].GetOpType()
            != UsdGeom.XformOp.TypeTranslate
        ):
            raise RuntimeError(
                f"endpoint transform stack is not translation-only: {path}"
            )
        xformable.AddRotateZOp().Set(math.degrees(value))
        records[role] = {
            "prim_path": path,
            "yaw_rad": value,
            "yaw_degrees": math.degrees(value),
        }
    return records


def _yaw_for_anchor(anchor_index: int, endpoint: str) -> float:
    """Cover distinct full-circle yaws across five anchors."""
    # Avoid using both -pi and +pi: they are the same physical orientation.
    loose_yaws = tuple(
        fraction * math.pi for fraction in (-0.80, -0.40, 0.0, 0.40, 0.80)
    )
    fixed_yaws = (
        -0.65 * math.pi,
        -0.25 * math.pi,
        0.15 * math.pi,
        0.55 * math.pi,
        0.95 * math.pi,
    )
    if endpoint == "loose_plug":
        return loose_yaws[anchor_index]
    if endpoint == "fixed_receptacle":
        return fixed_yaws[anchor_index]
    raise ValueError(f"unknown endpoint role {endpoint!r}")


def _non_authorization_record(contract):
    """Make the missing keyed-yaw boundary explicit in every report."""
    reasons = [
        "current_rgbd_runtime_estimates_mask_derived_xy_only",
        "current_proxy_has_no_unique_polarization_key_geometry",
        "yaw_symmetry_has_multiple_equivalent_hypotheses",
        "second_calibrated_key_view_is_unavailable",
        "object_target_transforms_are_unqualified_candidates",
    ]
    if contract.current_proxy_has_unique_polarization_key:
        raise RuntimeError("multisite contract unexpectedly claims a key")
    if contract.pose_control_current_authorized:
        raise RuntimeError(
            "multisite contract unexpectedly authorizes control"
        )
    return {
        "control_authorized": False,
        "full_6d": False,
        "keyed_orientation_observed": False,
        "rejection_reasons": reasons,
        "uses_truth_orientation_for_vision_pose": False,
        "yaw_observed": False,
    }


def _endpoint_capture_record(capture_metrics, role):
    endpoint = capture_metrics[role]
    mask_depth = endpoint["mask_depth"]
    return {
        "mask_depth": mask_depth,
        "mask_pixel_count": int(mask_depth["pixel_count"]),
        "passed": endpoint["passed"] is True,
        "ray_plane_xy_error_m": float(endpoint["xy_error_m"]),
        "semantic_ids": [
            int(value)
            for value in capture_metrics["endpoint_semantic_ids"][role]
        ],
        "semantic_mask_center": endpoint["semantic_mask_center"],
        "visible_fraction": float(mask_depth["visible_fraction"]),
    }


def _trial_passed(trial):
    """Apply only the five-anchor RGB-D contract's strict acceptance."""
    endpoint_records = (
        trial["endpoints"]["loose_plug"],
        trial["endpoints"]["fixed_receptacle"],
    )
    return bool(
        trial["capture_passed"]
        and all(item["passed"] for item in endpoint_records)
        and all(
            item["ray_plane_xy_error_m"]
            <= STRICT_MAXIMUM_XY_ERROR_M
            for item in endpoint_records
        )
        and all(
            item["semantic_ids"]
            and all(value not in (0, 1) for value in item["semantic_ids"])
            for item in endpoint_records
        )
        and trial["resource_cleanup"]["resources_released"] is True
        and trial["timeline_state"]["restored"] is True
        and trial["object_pose_writes_after_start"] == 0
        and trial["pose_scope"]["yaw_observed"] is False
        and trial["pose_scope"]["full_6d"] is False
        and trial["pose_scope"]["control_authorized"] is False
    )


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    output_dir = _repository_output_path(repository, arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": False,
        "gui": arguments.gui,
        "required_trial_count": 5,
        "strict_maximum_xy_error_m": STRICT_MAXIMUM_XY_ERROR_M,
        "trials": [],
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
            create_new_stage,
            get_current_stage,
        )
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from kcg_connector.d38999_multisite_vision6d import (
            load_d38999_multisite_vision6d_contract,
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

        config_path = Path(arguments.config).expanduser().resolve()
        rgbd_config_path = Path(arguments.rgbd_config).expanduser().resolve()
        contract = load_d38999_multisite_vision6d_contract(
            config_path, repository=repository
        )
        rgbd = load_rgbd_bootstrap(rgbd_config_path)
        tabletop = load_d38999_tabletop_scene(
            contract.input_paths["tabletop_scene"]
        )
        asset_path = verify_d38999_tabletop_asset(tabletop, repository)
        if Path(rgbd.tabletop_config).name != (
            contract.input_paths["tabletop_scene"].name
        ):
            raise ValueError("RGB-D and multi-position tabletop inputs differ")
        if len(contract.required_anchor_pairs) != 5:
            raise ValueError("multi-position RGB-D requires exactly 5 anchors")
        if rgbd.acceptance.maximum_xy_centroid_error_m != (
            STRICT_MAXIMUM_XY_ERROR_M
        ):
            raise ValueError("RGB-D XY acceptance must remain exactly 10 mm")
        report.update(
            {
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "rgbd_config_path": str(rgbd_config_path),
                "rgbd_config_sha256": _sha256(rgbd_config_path),
                "pose_scope": _non_authorization_record(contract),
            }
        )
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

        for anchor_index, anchor in enumerate(
            contract.required_anchor_pairs
        ):
            trial = {
                "trial_index": anchor_index + 1,
                "anchor_id": anchor["id"],
                "passed": False,
                "object_pose_writes_before_start": 4,
                "object_pose_writes_after_start": 0,
                "physics_started_after_endpoint_authoring": True,
                "pose_scope": _non_authorization_record(contract),
            }
            world = None
            try:
                # A new USD context stage prevents camera, lights, labels, or
                # rigid state from leaking across trials.
                World.clear_instance()
                create_new_stage()
                simulation_app.update()
                world = World(
                    stage_units_in_meters=1.0,
                    physics_dt=1.0 / tabletop.physics.rate_hz,
                    rendering_dt=1.0 / 60.0,
                    backend="numpy",
                    device="cpu",
                )
                stage = get_current_stage()
                trial_scene = _scene_for_anchor(tabletop, anchor)
                trial["authored_scene"] = author_d38999_tabletop_scene(
                    stage,
                    trial_scene,
                    asset_path,
                    add_reference_to_stage=add_reference_to_stage,
                    Gf=Gf,
                    Sdf=Sdf,
                    UsdGeom=UsdGeom,
                    UsdPhysics=UsdPhysics,
                    UsdShade=UsdShade,
                    physics_utils=physicsUtils,
                )
                loose_yaw = _yaw_for_anchor(anchor_index, "loose_plug")
                fixed_yaw = _yaw_for_anchor(
                    anchor_index, "fixed_receptacle"
                )
                trial["endpoint_authoring_before_physics"] = {
                    "positions": {
                        "loose_plug_xyz_m": list(
                            trial_scene.loose_endpoint.initial_origin_m
                        ),
                        "fixed_receptacle_xyz_m": list(
                            trial_scene.fixed_endpoint.receptacle_origin_m
                        ),
                        "fixed_fixture_center_xyz_m": list(
                            trial_scene.fixed_endpoint.fixture_center_m
                        ),
                    },
                    "yaws": _author_endpoint_yaws_before_physics(
                        stage=stage,
                        tabletop=trial_scene,
                        loose_yaw_rad=loose_yaw,
                        fixed_yaw_rad=fixed_yaw,
                        UsdGeom=UsdGeom,
                    ),
                }
                loose_prim = stage.GetPrimAtPath(
                    trial_scene.asset.loose_plug_prim_path
                )
                fixed_prim = stage.GetPrimAtPath(
                    trial_scene.asset.fixed_receptacle_prim_path
                )
                body = world.scene.add(
                    SingleRigidPrim(
                        prim_path=trial_scene.asset.body_prim_path,
                        name=f"multisite_loose_body_{anchor_index + 1}",
                    )
                )

                # Physics begins here.  There are no transform-authoring calls
                # after this reset; only stepping, readback, render capture,
                # and owned render-resource cleanup remain.
                world.reset()
                world.get_physics_context().set_gravity(
                    trial_scene.physics.gravity_m_s2
                )
                for _ in range(trial_scene.physics.settle_steps):
                    world.step(render=True)

                trial_output = output_dir / (
                    f"trial_{anchor_index + 1:02d}_{anchor['id']}"
                )
                capture = capture_d38999_rgbd_runtime(
                    bindings=runtime_bindings,
                    simulation_app=simulation_app,
                    world=world,
                    stage=stage,
                    tabletop=trial_scene,
                    rgbd=rgbd,
                    loose_prim=loose_prim,
                    fixed_prim=fixed_prim,
                    body=body,
                    output_dir=trial_output,
                )
                trial["capture_passed"] = capture.passed
                trial["camera_projection"] = capture.metrics.get(
                    "camera_projection"
                )
                trial["endpoints"] = {
                    role: _endpoint_capture_record(capture.metrics, role)
                    for role in ("loose_plug", "fixed_receptacle")
                }
                trial["observed_semantic_ids"] = capture.metrics.get(
                    "observed_semantic_ids"
                )
                trial["resource_cleanup"] = capture.metrics.get(
                    "resource_cleanup"
                )
                trial["timeline_pause"] = capture.metrics.get(
                    "timeline_pause"
                )
                trial["timeline_state"] = capture.metrics.get(
                    "timeline_state"
                )
                trial["passed"] = _trial_passed(trial)
                if not trial["passed"]:
                    trial["error"] = capture.metrics.get(
                        "error", "strict five-anchor acceptance failed"
                    )
            except BaseException as exception:
                trial.update(
                    {
                        "error": f"{type(exception).__name__}: {exception}",
                        "passed": False,
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                if world is not None:
                    try:
                        world.stop()
                        simulation_app.update()
                    except BaseException as exception:
                        trial["world_stop_error"] = (
                            f"{type(exception).__name__}: {exception}"
                        )
                        trial["passed"] = False
                World.clear_instance()
            report["trials"].append(trial)
            print(
                json.dumps(trial, allow_nan=False, sort_keys=True),
                flush=True,
            )

        passed = bool(
            len(report["trials"]) == report["required_trial_count"]
            and all(trial["passed"] for trial in report["trials"])
            and report["pose_scope"]["yaw_observed"] is False
            and report["pose_scope"]["full_6d"] is False
            and report["pose_scope"]["control_authorized"] is False
        )
        report["passed_trial_count"] = sum(
            1 for trial in report["trials"] if trial["passed"]
        )
        report["passed"] = passed
    except BaseException as exception:
        report.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        report_path = output_dir / "report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        print(
            "ISAAC D38999 MULTISITE RGBD V1 "
            + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
