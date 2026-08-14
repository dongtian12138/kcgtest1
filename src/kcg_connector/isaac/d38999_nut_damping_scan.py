#!/usr/bin/env python3

"""Measure in-memory rigid-body angular damping candidates headlessly."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback


def _gf_world_pose(Gf, Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _gf_quaternion_error_radians(first, second):
    relative = first.GetInverse() * second
    real = max(-1.0, min(1.0, abs(float(relative.GetReal()))))
    return 2.0 * math.acos(real)


def _gf_quaternion_finite(value):
    imaginary = value.GetImaginary()
    return all(
        math.isfinite(item)
        for item in (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    )


def _quaternion_z_axis(value):
    import numpy as np

    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return np.full(3, np.nan, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=np.float64,
    )


def _tilt_from_world_z(value):
    axis = _quaternion_z_axis(value)
    if not all(math.isfinite(float(item)) for item in axis):
        return float("inf")
    return math.acos(max(-1.0, min(1.0, float(axis[2]))))


def _arguments(repository):
    parser = argparse.ArgumentParser(
        description="Scan D38999 coupling-nut rigid-body angular damping"
    )
    parser.add_argument(
        "--scan-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_nut_damping_scan_v1.yaml"
        ),
    )
    parser.add_argument(
        "--scene-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="optional explicit JSON path; default is a timestamped artifact",
    )
    return parser.parse_args()


def _default_output(repository):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        repository
        / "artifacts/kcg_connector/isaac/d38999_nut_damping_scan_v1"
        / f"scan_{timestamp}.json"
    )


def _run_repeat(
    *,
    simulation_app,
    candidate,
    scan_config,
    scene_config,
    asset_path,
    runtime,
):
    import numpy as np

    World = runtime["World"]
    SingleRigidPrim = runtime["SingleRigidPrim"]
    add_reference_to_stage = runtime["add_reference_to_stage"]
    get_current_stage = runtime["get_current_stage"]
    physicsUtils = runtime["physicsUtils"]
    omni_usd = runtime["omni_usd"]
    Gf = runtime["Gf"]
    PhysxSchema = runtime["PhysxSchema"]
    Sdf = runtime["Sdf"]
    Usd = runtime["Usd"]
    UsdGeom = runtime["UsdGeom"]
    UsdPhysics = runtime["UsdPhysics"]
    UsdShade = runtime["UsdShade"]
    author_scene = runtime["author_scene"]

    World.clear_instance()
    omni_usd.get_context().new_stage()
    simulation_app.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / scan_config.experiment.physics_rate_hz,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    author_metrics = author_scene(
        stage,
        scene_config,
        asset_path,
        add_reference_to_stage=add_reference_to_stage,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        physics_utils=physicsUtils,
    )
    nut_prim = stage.GetPrimAtPath(scene_config.asset.nut_prim_path)
    fixed_prim = stage.GetPrimAtPath(
        scene_config.asset.fixed_receptacle_prim_path
    )
    if not nut_prim.IsValid() or not fixed_prim.IsValid():
        raise RuntimeError("D38999 scan prims are missing")

    api_preexisting = nut_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
    authored_damping = None
    if not candidate.is_baseline:
        if api_preexisting:
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(nut_prim)
        else:
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(nut_prim)
        rigid_api.CreateAngularDampingAttr().Set(
            candidate.angular_damping
        )
        authored_damping = float(
            rigid_api.GetAngularDampingAttr().Get()
        )
        if not math.isclose(
            authored_damping,
            candidate.expected_resolved_angular_damping,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise RuntimeError("authored angular damping did not resolve")

    body = world.scene.add(
        SingleRigidPrim(
            prim_path=scene_config.asset.body_prim_path,
            name=f"scan_body_{candidate.run_order}",
        )
    )
    nut = world.scene.add(
        SingleRigidPrim(
            prim_path=scene_config.asset.nut_prim_path,
            name=f"scan_nut_{candidate.run_order}",
        )
    )
    world.reset()
    world.get_physics_context().set_gravity(
        scene_config.physics.gravity_m_s2
    )

    initial_body_position, _ = body.get_world_pose()
    initial_nut_position, _ = nut.get_world_pose()
    initial_center = (
        2.0 * np.asarray(initial_body_position, dtype=np.float64)
        + np.asarray(initial_nut_position, dtype=np.float64)
    ) / 3.0
    fixed_initial_position, fixed_initial_orientation = _gf_world_pose(
        Gf, Usd, UsdGeom, fixed_prim
    )

    finite_throughout = True
    maximum_transient_penetration = 0.0
    maximum_xy_drift = 0.0
    maximum_axis_tilt = 0.0
    maximum_fixed_translation_drift = 0.0
    maximum_fixed_rotation_drift = 0.0
    maximum_tail_displacement = 0.0
    maximum_tail_linear_speed = 0.0
    maximum_tail_body_angular_speed = 0.0
    maximum_tail_nut_angular_speed = 0.0
    maximum_tail_relative_axis_speed = 0.0
    tail_start_center = None

    for step_index in range(scan_config.experiment.settle_steps):
        world.step(render=False)
        body_position, body_orientation = body.get_world_pose()
        nut_position, nut_orientation = nut.get_world_pose()
        body_position = np.asarray(body_position, dtype=np.float64)
        nut_position = np.asarray(nut_position, dtype=np.float64)
        body_orientation = np.asarray(body_orientation, dtype=np.float64)
        nut_orientation = np.asarray(nut_orientation, dtype=np.float64)
        body_linear = np.asarray(
            body.get_linear_velocity(), dtype=np.float64
        )
        nut_linear = np.asarray(
            nut.get_linear_velocity(), dtype=np.float64
        )
        body_angular = np.asarray(
            body.get_angular_velocity(), dtype=np.float64
        )
        nut_angular = np.asarray(
            nut.get_angular_velocity(), dtype=np.float64
        )
        current_center = (2.0 * body_position + nut_position) / 3.0
        fixed_position, fixed_orientation = _gf_world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        fixed_position = np.asarray(fixed_position, dtype=np.float64)
        values = np.concatenate(
            (
                body_position,
                body_orientation,
                nut_position,
                nut_orientation,
                body_linear,
                nut_linear,
                body_angular,
                nut_angular,
                fixed_position,
            )
        )
        finite_throughout = bool(
            finite_throughout
            and np.all(np.isfinite(values))
            and _gf_quaternion_finite(fixed_orientation)
        )

        current_bottom = min(
            float(body_position[2])
            + scene_config.loose_endpoint.body_bottom_offset_m,
            float(nut_position[2])
            + scene_config.loose_endpoint.nut_bottom_offset_m,
        )
        maximum_transient_penetration = max(
            maximum_transient_penetration,
            scene_config.table.top_z_m - current_bottom,
        )
        maximum_xy_drift = max(
            maximum_xy_drift,
            float(
                np.linalg.norm(current_center[:2] - initial_center[:2])
            ),
        )
        maximum_axis_tilt = max(
            maximum_axis_tilt,
            _tilt_from_world_z(body_orientation),
            _tilt_from_world_z(nut_orientation),
        )
        maximum_fixed_translation_drift = max(
            maximum_fixed_translation_drift,
            float(
                np.linalg.norm(
                    fixed_position
                    - np.asarray(
                        fixed_initial_position, dtype=np.float64
                    )
                )
            ),
        )
        maximum_fixed_rotation_drift = max(
            maximum_fixed_rotation_drift,
            _gf_quaternion_error_radians(
                fixed_initial_orientation, fixed_orientation
            ),
        )

        if step_index == (
            scan_config.experiment.settle_steps
            - scan_config.experiment.tail_steps
            - 1
        ):
            tail_start_center = current_center.copy()
        if tail_start_center is not None:
            body_speed = float(np.linalg.norm(body_angular))
            nut_speed = float(np.linalg.norm(nut_angular))
            body_axis = _quaternion_z_axis(body_orientation)
            relative_axis_speed = abs(
                float(np.dot(nut_angular - body_angular, body_axis))
            )
            maximum_tail_displacement = max(
                maximum_tail_displacement,
                float(np.linalg.norm(current_center - tail_start_center)),
            )
            maximum_tail_linear_speed = max(
                maximum_tail_linear_speed,
                float(np.linalg.norm(body_linear)),
                float(np.linalg.norm(nut_linear)),
            )
            maximum_tail_body_angular_speed = max(
                maximum_tail_body_angular_speed, body_speed
            )
            maximum_tail_nut_angular_speed = max(
                maximum_tail_nut_angular_speed, nut_speed
            )
            maximum_tail_relative_axis_speed = max(
                maximum_tail_relative_axis_speed, relative_axis_speed
            )

    final_body_position, _ = body.get_world_pose()
    final_nut_position, _ = nut.get_world_pose()
    final_body_position = np.asarray(final_body_position, dtype=np.float64)
    final_nut_position = np.asarray(final_nut_position, dtype=np.float64)
    final_center = (
        2.0 * final_body_position + final_nut_position
    ) / 3.0
    if tail_start_center is None:
        raise RuntimeError("D38999 damping tail window was not sampled")
    final_bottom = min(
        float(final_body_position[2])
        + scene_config.loose_endpoint.body_bottom_offset_m,
        float(final_nut_position[2])
        + scene_config.loose_endpoint.nut_bottom_offset_m,
    )
    final_surface_error = final_bottom - scene_config.table.top_z_m
    vertical_drop = float(initial_center[2] - final_center[2])

    dropped = bool(
        scene_config.physics.minimum_vertical_drop_m
        <= vertical_drop
        <= scene_config.physics.maximum_vertical_drop_m
    )
    penetration_safe = bool(
        maximum_transient_penetration
        <= scene_config.physics.maximum_transient_table_penetration_m
    )
    on_surface = bool(
        -scene_config.physics.maximum_transient_table_penetration_m
        <= final_surface_error
        <= scene_config.physics.maximum_final_surface_gap_m
    )
    drift_safe = bool(
        maximum_xy_drift <= scene_config.physics.maximum_xy_drift_m
    )
    tilt_safe = bool(
        maximum_axis_tilt
        <= scene_config.physics.maximum_upright_axis_tilt_rad
    )
    tail_scene_safe = bool(
        maximum_tail_displacement
        <= scene_config.physics.maximum_tail_displacement_m
        and maximum_tail_linear_speed
        <= scene_config.physics.maximum_tail_linear_speed_m_s
        and max(
            maximum_tail_body_angular_speed,
            maximum_tail_nut_angular_speed,
        )
        <= scene_config.physics.maximum_tail_angular_speed_rad_s
    )
    fixed_safe = bool(
        maximum_fixed_translation_drift
        <= scene_config.physics.maximum_fixed_translation_drift_m
        and maximum_fixed_rotation_drift
        <= scene_config.physics.maximum_fixed_rotation_drift_rad
    )
    scene_safety_pass = bool(
        finite_throughout
        and dropped
        and penetration_safe
        and on_surface
        and drift_safe
        and tilt_safe
        and tail_scene_safe
        and fixed_safe
        and scan_config.experiment.object_pose_writes_after_start == 0
    )
    result = {
        "api_preexisting_before_override": api_preexisting,
        "author_metrics": author_metrics,
        "authored_angular_damping": authored_damping,
        "candidate_id": candidate.candidate_id,
        "expected_resolved_angular_damping": (
            candidate.expected_resolved_angular_damping
        ),
        "finite_throughout": finite_throughout,
        "fixed_endpoint_safe": fixed_safe,
        "maximum_axis_tilt_rad": maximum_axis_tilt,
        "maximum_fixed_rotation_drift_rad": (
            maximum_fixed_rotation_drift
        ),
        "maximum_fixed_translation_drift_m": (
            maximum_fixed_translation_drift
        ),
        "maximum_tail_body_angular_speed_rad_s": (
            maximum_tail_body_angular_speed
        ),
        "maximum_tail_displacement_m": maximum_tail_displacement,
        "maximum_tail_linear_speed_m_s": maximum_tail_linear_speed,
        "maximum_tail_nut_angular_speed_rad_s": (
            maximum_tail_nut_angular_speed
        ),
        "maximum_tail_relative_axis_speed_rad_s": (
            maximum_tail_relative_axis_speed
        ),
        "maximum_transient_penetration_m": max(
            0.0, maximum_transient_penetration
        ),
        "maximum_xy_drift_m": maximum_xy_drift,
        "mechanism": candidate.mechanism,
        "object_pose_writes_after_start": 0,
        "on_table_surface": on_surface,
        "scene_safety_pass": scene_safety_pass,
        "source_asset_mutated": False,
        "tail_scene_safe": tail_scene_safe,
        "vertical_drop_m": vertical_drop,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _candidate_summary(candidate, repeats):
    return {
        "candidate_id": candidate.candidate_id,
        "repeat_count": len(repeats),
        "every_repeat_finite": all(
            item["finite_throughout"] for item in repeats
        ),
        "every_repeat_scene_safety_pass": all(
            item["scene_safety_pass"] for item in repeats
        ),
        "maximum_tail_scene_angular_speed_rad_s": max(
            max(
                item["maximum_tail_body_angular_speed_rad_s"],
                item["maximum_tail_nut_angular_speed_rad_s"],
            )
            for item in repeats
        ),
        "maximum_tail_nut_angular_speed_rad_s": max(
            item["maximum_tail_nut_angular_speed_rad_s"]
            for item in repeats
        ),
        "maximum_tail_relative_axis_speed_rad_s": max(
            item["maximum_tail_relative_axis_speed_rad_s"]
            for item in repeats
        ),
    }


def _process_exit_code(passed):
    """Map the completed scan verdict to a fail-closed CLI exit code."""
    if type(passed) is not bool:
        raise TypeError("passed must be boolean")
    return 0 if passed else 1


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    report = {
        "automatic_promotion_permitted": False,
        "object_pose_writes_after_start": 0,
        "passed": False,
        "scan": "kcg_d38999_nut_damping_scan_v1",
        "source_asset_mutated": False,
    }
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        import omni.usd
        from omni.physx.scripts import physicsUtils
        from pxr import (
            Gf,
            PhysxSchema,
            Sdf,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdShade,
        )

        from kcg_connector.d38999_nut_damping_scan import (
            load_d38999_nut_damping_scan,
            select_damping_candidate,
        )
        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
            load_d38999_tabletop_scene,
            verify_d38999_tabletop_asset,
        )

        scan_config = load_d38999_nut_damping_scan(
            Path(arguments.scan_config).expanduser().resolve()
        )
        scene_config = load_d38999_tabletop_scene(
            Path(arguments.scene_config).expanduser().resolve()
        )
        if (
            scan_config.scope.scene_schema_version
            != scene_config.schema_version
        ):
            raise ValueError("scan and scene schema versions do not match")
        if scan_config.scope.asset_sha256 != scene_config.asset.sha256:
            raise ValueError("scan and scene asset hashes do not match")
        if (
            scan_config.experiment.physics_rate_hz
            != scene_config.physics.rate_hz
        ):
            raise ValueError("scan and scene physics rates do not match")
        asset_path = verify_d38999_tabletop_asset(
            scene_config, repository
        )
        source_hash_before = hashlib.sha256(
            asset_path.read_bytes()
        ).hexdigest()

        runtime = {
            "Gf": Gf,
            "PhysxSchema": PhysxSchema,
            "Sdf": Sdf,
            "SingleRigidPrim": SingleRigidPrim,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "UsdPhysics": UsdPhysics,
            "UsdShade": UsdShade,
            "World": World,
            "add_reference_to_stage": add_reference_to_stage,
            "author_scene": author_d38999_tabletop_scene,
            "get_current_stage": get_current_stage,
            "omni_usd": omni.usd,
            "physicsUtils": physicsUtils,
        }
        candidate_results = []
        summaries = []
        for candidate in scan_config.candidates:
            repeats = []
            for repeat_index in range(
                scan_config.experiment.repeats_per_candidate
            ):
                result = _run_repeat(
                    simulation_app=simulation_app,
                    candidate=candidate,
                    scan_config=scan_config,
                    scene_config=scene_config,
                    asset_path=asset_path,
                    runtime=runtime,
                )
                result["repeat_index"] = repeat_index
                repeats.append(result)
                print(
                    "D38999 NUT DAMPING SAMPLE "
                    + json.dumps(
                        {
                            "candidate_id": candidate.candidate_id,
                            "repeat_index": repeat_index,
                            "scene_safety_pass": result[
                                "scene_safety_pass"
                            ],
                            "tail_nut_rad_s": result[
                                "maximum_tail_nut_angular_speed_rad_s"
                            ],
                            "tail_relative_axis_rad_s": result[
                                "maximum_tail_relative_axis_speed_rad_s"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            candidate_results.append(
                {
                    "candidate": asdict_candidate(candidate),
                    "repeats": repeats,
                }
            )
            summaries.append(_candidate_summary(candidate, repeats))

        selection = select_damping_candidate(scan_config, summaries)
        source_hash_after = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        source_asset_mutated = bool(source_hash_after != source_hash_before)
        passed = bool(
            selection.baseline_valid
            and selection.selected_candidate_id is not None
            and not source_asset_mutated
        )
        report.update(
            {
                "candidate_results": candidate_results,
                "candidate_summaries": summaries,
                "config": scan_config.as_dict(),
                "passed": passed,
                "scene_config_sha256": scene_config.asset.sha256,
                "selection": selection.as_dict(),
                "source_asset_mutated": source_asset_mutated,
                "source_asset_sha256_after": source_hash_after,
                "source_asset_sha256_before": source_hash_before,
            }
        )
        output_path = (
            Path(arguments.output).expanduser().resolve()
            if arguments.output
            else _default_output(repository)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        report["artifact_path"] = str(output_path)
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        print(
            "ISAAC D38999 NUT DAMPING SCAN V1 "
            + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        report.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
            }
        )
        traceback.print_exc()
        print(json.dumps(report, sort_keys=True), flush=True)
        print("ISAAC D38999 NUT DAMPING SCAN V1 FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=_process_exit_code(passed))
    return _process_exit_code(passed)


def asdict_candidate(candidate):
    """Return the finite public candidate fields without dataclass imports."""
    return {
        "angular_damping": candidate.angular_damping,
        "candidate_id": candidate.candidate_id,
        "efficacy_status": candidate.efficacy_status,
        "expected_resolved_angular_damping": (
            candidate.expected_resolved_angular_damping
        ),
        "mechanism": candidate.mechanism,
        "requires_articulation": candidate.requires_articulation,
        "run_order": candidate.run_order,
        "target_component": candidate.target_component,
    }


if __name__ == "__main__":
    raise SystemExit(main())
