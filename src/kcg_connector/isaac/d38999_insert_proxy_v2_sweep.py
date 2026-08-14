#!/usr/bin/env python3

"""PhysX tolerance sweep for the D38999 insertion proxy V2.

The actuator assigns one known initial axial velocity and then free-coasts. It
consumes no feedback: contact identity, contact normals, object pose, and
object velocity never steer the motion.  Rigid-body states are
sampled only after each physics step for post-hoc scoring and wrench estimates.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

from kcg_connector.d38999_insert_proxy_v2 import (
    DEFAULT_CONFIG_PATH,
    load_insert_proxy_v2,
)


def _arguments():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/d38999_insert_proxy_v2.usda"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_insert_proxy_v2"
        ),
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("tolerance sweep requires explicit --run")
    if args.keep_open and not args.gui:
        parser.error("--keep-open requires --gui")
    return args


def _trial_matrix(config):
    sweep = config.document["tolerance_sweep"]
    trials = [
        {
            "trial_id": "nominal",
            "sweep_axis": "nominal",
            "x_offset_m": 0.0,
            "y_offset_m": 0.0,
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "yaw_rad": 0.0,
        }
    ]
    fields = (
        ("x_offset_m", "x_offset_m"),
        ("y_offset_m", "y_offset_m"),
        ("tilt_x_rad", "tilt_x_rad"),
        ("tilt_y_rad", "tilt_y_rad"),
        ("yaw_rad", "yaw_rad"),
    )
    for field, sweep_axis in fields:
        for value in sweep[field]:
            value = float(value)
            if abs(value) < 1.0e-15:
                continue
            trial = {
                "trial_id": f"{sweep_axis}_{value:+.9f}".replace("+", "p").replace("-", "m"),
                "sweep_axis": sweep_axis,
                "x_offset_m": 0.0,
                "y_offset_m": 0.0,
                "tilt_x_rad": 0.0,
                "tilt_y_rad": 0.0,
                "yaw_rad": 0.0,
            }
            trial[field] = value
            trials.append(trial)
    return trials


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _write_visualization(output_dir, results):
    from PIL import Image, ImageDraw

    width, height = 1200, 700
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), "D38999 insertion proxy V2 tolerance sweep", fill=(20, 30, 45))
    axes = ["x_offset_m", "y_offset_m", "tilt_x_rad", "tilt_y_rad", "yaw_rad"]
    colors = {True: (44, 150, 82), False: (205, 65, 64)}
    top = 70
    band = 118
    for row, axis in enumerate(axes):
        subset = [item for item in results if item["sweep_axis"] == axis]
        subset.append(next(item for item in results if item["trial_id"] == "nominal"))
        subset.sort(key=lambda item: item[axis])
        y = top + row * band
        draw.text((24, y), axis, fill=(25, 35, 50))
        if not subset:
            continue
        minimum = min(item[axis] for item in subset)
        maximum = max(item[axis] for item in subset)
        span = max(maximum - minimum, 1.0e-12)
        draw.line((210, y + 24, 1140, y + 24), fill=(120, 130, 145), width=2)
        for item in subset:
            x = 210 + int(930 * (item[axis] - minimum) / span)
            color = colors[bool(item["success"])]
            draw.ellipse((x - 7, y + 17, x + 7, y + 31), fill=color, outline=(25, 25, 25))
        draw.text((210, y + 42), f"min {minimum:.6g}", fill=(70, 80, 95))
        draw.text((1010, y + 42), f"max {maximum:.6g}", fill=(70, 80, 95))
    image.save(output_dir / "tolerance_sweep.png")


def main():
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[3]
    config = load_insert_proxy_v2(arguments.config)
    asset_path = Path(arguments.asset).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    trials = _trial_matrix(config)

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
        "schema_version": "kcg_d38999_insert_proxy_v2_tolerance_sweep_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "asset_path": str(asset_path),
        "asset_sha256": _sha256(asset_path),
        "controller_inputs": [],
        "initial_actuation": {
            "kind": "one_shot_world_linear_velocity",
            "velocity_m_s": [0.0, 0.0, 0.004],
        },
        "controller_truth_inputs": [],
        "physx_contact_normal_used_for_control": False,
        "collider_identity_used_for_control": False,
        "truth_scope": "post_hoc_scoring_only",
        "trial_count": len(trials),
        "passed": False,
    }
    try:
        import numpy as np

        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        World.clear_instance()
        omni.usd.get_context().new_stage()
        simulation_app.update()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 240.0,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Trials")
        bodies = []
        nuts = []
        base_positions = []
        for index, trial in enumerate(trials):
            row, column = divmod(index, 8)
            base = np.asarray((column * 0.080, row * 0.080, 0.0), dtype=np.float64)
            base_positions.append(base)
            trial_path = f"/World/Trials/Trial_{index:03d}"
            root = UsdGeom.Xform.Define(stage, trial_path)
            root.GetPrim().GetReferences().AddReference(
                str(asset_path), "/World/D38999InsertProxyV2"
            )
            UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*base))
            plug_prim = stage.OverridePrim(trial_path + "/Plug")
            plug_xform = UsdGeom.Xformable(plug_prim)
            plug_xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    trial["x_offset_m"],
                    trial["y_offset_m"],
                    -config.preinsert_gap,
                )
            )
            plug_xform.AddRotateXYZOp().Set(
                Gf.Vec3f(
                    math.degrees(trial["tilt_x_rad"]),
                    math.degrees(trial["tilt_y_rad"]),
                    math.degrees(trial["yaw_rad"]),
                )
            )
            body_path = trial_path + "/Plug/Body"
            nut_path = trial_path + "/Plug/CouplingNut"
            bodies.append(
                world.scene.add(SingleRigidPrim(prim_path=body_path, name=f"body_{index:03d}"))
            )
            nuts.append(
                world.scene.add(SingleRigidPrim(prim_path=nut_path, name=f"nut_{index:03d}"))
            )

        world.reset()
        world.get_physics_context().set_gravity(0.0)
        initial_positions = np.asarray(
            [body.get_world_pose()[0] for body in bodies], dtype=np.float64
        )
        target_speed = 0.004
        for body, nut in zip(bodies, nuts):
            body.set_linear_velocity((0.0, 0.0, target_speed))
            nut.set_linear_velocity((0.0, 0.0, target_speed))
        previous_body_velocity = np.asarray(
            [body.get_linear_velocity() for body in bodies], dtype=np.float64
        )
        previous_nut_velocity = np.asarray(
            [nut.get_linear_velocity() for nut in nuts], dtype=np.float64
        )
        previous_body_angular = np.asarray(
            [body.get_angular_velocity() for body in bodies], dtype=np.float64
        )
        previous_nut_angular = np.asarray(
            [nut.get_angular_velocity() for nut in nuts], dtype=np.float64
        )
        records = [
            {
                **trial,
                "peak_axial_force_n": 0.0,
                "peak_lateral_force_n": 0.0,
                "peak_bending_moment_nm": 0.0,
                "maximum_progress_m": 0.0,
                "contact_samples": 0,
            }
            for trial in trials
        ]
        dt = 1.0 / 240.0
        step_count = int(math.ceil(6.5 / dt))
        body_mass = config.physics.plug_body_mass_kg
        nut_mass = config.physics.coupling_nut_mass_kg
        body_inertia = np.asarray(
            config.physics.plug_body_diagonal_inertia_kg_m2, dtype=np.float64
        )
        nut_inertia = np.asarray(
            config.physics.coupling_nut_diagonal_inertia_kg_m2, dtype=np.float64
        )
        for step_index in range(step_count):
            applied_z = np.zeros(len(trials), dtype=np.float64)
            world.step(render=arguments.gui and step_index % 4 == 0)
            new_body_velocity = np.asarray(
                [body.get_linear_velocity() for body in bodies], dtype=np.float64
            )
            new_nut_velocity = np.asarray(
                [nut.get_linear_velocity() for nut in nuts], dtype=np.float64
            )
            new_body_angular = np.asarray(
                [body.get_angular_velocity() for body in bodies], dtype=np.float64
            )
            new_nut_angular = np.asarray(
                [nut.get_angular_velocity() for nut in nuts], dtype=np.float64
            )
            combined_force = (
                body_mass * (new_body_velocity - previous_body_velocity) / dt
                + nut_mass * (new_nut_velocity - previous_nut_velocity) / dt
            )
            combined_force[:, 2] -= applied_z
            combined_moment = (
                (new_body_angular - previous_body_angular) / dt * body_inertia
                + (new_nut_angular - previous_nut_angular) / dt * nut_inertia
            )
            positions = np.asarray(
                [body.get_world_pose()[0] for body in bodies], dtype=np.float64
            )
            progress = positions[:, 2] - initial_positions[:, 2]
            for index, record in enumerate(records):
                axial = abs(float(combined_force[index, 2]))
                lateral = float(np.linalg.norm(combined_force[index, :2]))
                bending = float(np.linalg.norm(combined_moment[index, :2]))
                # Ignore the first reset transient; thereafter this is
                # evaluation only and never feeds the velocity servo.
                if step_index > 4:
                    record["peak_axial_force_n"] = max(record["peak_axial_force_n"], axial)
                    record["peak_lateral_force_n"] = max(record["peak_lateral_force_n"], lateral)
                    record["peak_bending_moment_nm"] = max(
                        record["peak_bending_moment_nm"], bending
                    )
                    if axial > 0.20 or lateral > 0.20:
                        record["contact_samples"] += 1
                record["maximum_progress_m"] = max(
                    record["maximum_progress_m"], float(progress[index])
                )
            previous_body_velocity = new_body_velocity
            previous_nut_velocity = new_nut_velocity
            previous_body_angular = new_body_angular
            previous_nut_angular = new_nut_angular

        success_progress = config.preinsert_gap + config.insertion_depth - 0.00050
        guide_progress = config.preinsert_gap + config.receptacle.entrance_chamfer_length
        for record in records:
            progress = record["maximum_progress_m"]
            record["entered_guide"] = bool(progress >= guide_progress)
            record["success"] = bool(progress >= success_progress)
            record["insertion_depth_m"] = max(0.0, progress - config.preinsert_gap)
            if record["success"]:
                record["failure_type"] = "NONE"
            elif progress < config.preinsert_gap + 0.0005:
                record["failure_type"] = "ENTRY_BLOCKED"
            elif progress < guide_progress:
                record["failure_type"] = "CHAMFER_JAM"
            else:
                record["failure_type"] = "GUIDE_OR_KEY_JAM"

        boundaries = {}
        for axis in ("x_offset_m", "y_offset_m", "tilt_x_rad", "tilt_y_rad", "yaw_rad"):
            subset = [item for item in records if item["sweep_axis"] == axis]
            positive_success = [item[axis] for item in subset if item["success"] and item[axis] > 0]
            negative_success = [abs(item[axis]) for item in subset if item["success"] and item[axis] < 0]
            positive_failure = [item[axis] for item in subset if not item["success"] and item[axis] > 0]
            negative_failure = [abs(item[axis]) for item in subset if not item["success"] and item[axis] < 0]
            measured = min(
                max(positive_success, default=0.0),
                max(negative_success, default=max(positive_success, default=0.0)),
            )
            if axis == "yaw_rad":
                measured = max((abs(item[axis]) for item in subset if item["success"]), default=0.0)
            bracketed = bool(
                measured > 0.0
                and any(value > measured for value in positive_failure)
                and any(value > measured for value in negative_failure)
            )
            boundaries[axis] = {
                "measured_success_boundary_abs": float(measured),
                "nearest_failure_abs": min(
                    [
                        value
                        for value in positive_failure + negative_failure
                        if value > measured
                    ],
                    default=None,
                ),
                "failure_bracketed_both_signs": bracketed,
                "authorization_gate_abs": float(
                    config.gate_fraction * measured if bracketed else 0.0
                ),
                "gate_fraction": config.gate_fraction,
            }
        report.update(
            {
                "passed": bool(next(item for item in records if item["trial_id"] == "nominal")["success"]),
                "nominal_success": bool(next(item for item in records if item["trial_id"] == "nominal")["success"]),
                "success_count": sum(int(item["success"]) for item in records),
                "failure_count": sum(int(not item["success"]) for item in records),
                "measured_boundaries": boundaries,
                "results": records,
            }
        )
        (output_dir / "tolerance_sweep.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fields = list(records[0])
        with (output_dir / "tolerance_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        geometry_report = f"""# D38999 insertion proxy V2 geometry report

- Asset: `{asset_path}`
- Asset SHA-256: `{report['asset_sha256']}`
- Explicit insertion axis: connector assembly frame +Z
- Entry chamfer: {config.receptacle.entrance_chamfer_length * 1e3:.3f} mm, four collision slices
- Radial clearance: {config.radial_clearance * 1e3:.3f} mm
- Guide length: {config.receptacle.guide_length * 1e3:.3f} mm
- Insertion depth: {config.insertion_depth * 1e3:.3f} mm
- Symmetry/keying: C2, two opposing keys and two opposing collision channels
- Plug body mass: {config.physics.plug_body_mass_kg:.3f} kg
- Coupling nut mass: {config.physics.coupling_nut_mass_kg:.3f} kg
- Contact stiffness: {config.physics.compliant_contact_stiffness_n_m:.1f} N/m
- Contact damping: {config.physics.compliant_contact_damping_n_s_m:.1f} N s/m
- Thread label: **PROXY THREAD**; fine thread teeth are not modeled.
- Lock label: **PROXY LOCK**; final lock is analytic and not a real D38999 certification.
"""
        (output_dir / "geometry_report.md").write_text(geometry_report, encoding="utf-8")
        tolerance_report = f"""# D38999 insertion proxy V2 tolerance report

PhysX trials: {len(records)}; success: {report['success_count']}; failure: {report['failure_count']}.
The actuator assigned one known 4 mm/s initial axial velocity and then free-coasted.
Object truth, measured velocity, contact normals, collider identity, and contact
points were not controller inputs. State samples were used only for post-hoc progress,
wrench estimation, and failure classification.

Measured conservative gates are 65% of the observed axis-wise success boundary:

```json
{json.dumps(boundaries, indent=2, sort_keys=True)}
```
"""
        (output_dir / "tolerance_report.md").write_text(tolerance_report, encoding="utf-8")
        _write_visualization(output_dir, records)
        print(json.dumps({key: report[key] for key in ("passed", "trial_count", "success_count", "failure_count", "measured_boundaries")}, sort_keys=True))
        print("ISAAC D38999 INSERT PROXY V2 TOLERANCE SWEEP " + ("PASSED" if report["passed"] else "FAILED"))
        passed = bool(report["passed"])
        if arguments.keep_open:
            while simulation_app.is_running():
                world.step(render=True)
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        (output_dir / "failed_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        traceback.print_exc()
        print("ISAAC D38999 INSERT PROXY V2 TOLERANCE SWEEP FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
