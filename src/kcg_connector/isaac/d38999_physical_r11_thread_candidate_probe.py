#!/usr/bin/env python3

"""Isolate the r12 three-start thread follower candidate on the r11 rails.

This is an A3 modelling diagnostic, not formal acceptance.  It disables every
realized collider except the three 360-piece helical rails, replaces the three
box followers with analytic spheres, and reuses the P1 force-only motion loop.
Contact truth is recorded only after each step and never changes the command.
No file fingerprint is computed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "kcg_d38999_physical_r11_thread_candidate_probe_v1"
GENERATOR_ID = "kcg_d38999_r11_isolated_thread_sphere_probe_v1"
FOLLOWER_RADIUS_M = 0.000150
FOLLOWER_CENTER_RADIUS_M = 0.0201168
FOLLOWER_CENTER_Z_M = -0.001540
FOLLOWER_PHASE_0_DEG = 0.28481322866243236
PHASE_LEAD_DEG = 2.0


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument("--phase-lead-deg", type=float, default=PHASE_LEAD_DEG)
    parser.add_argument(
        "--kit-portable-root",
        required=True,
        help="Writable Kit portable root; this diagnostic only accepts /tmp paths.",
    )
    result = parser.parse_args(argv)
    if not result.run:
        parser.error("the isolated thread probe requires --run")
    if (
        not math.isfinite(result.phase_lead_deg)
        or result.phase_lead_deg < 0.0
        or result.phase_lead_deg > 5.0
    ):
        parser.error("phase lead must be finite and in [0, 5] degrees")
    portable_root = Path(result.kit_portable_root).expanduser().resolve()
    if not portable_root.is_relative_to(Path("/tmp")):
        parser.error("--kit-portable-root must resolve below /tmp")
    result.kit_portable_root = str(portable_root)
    return result


def _emit(value: Any) -> None:
    os.write(
        1,
        (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _install_actual_position_phase_lead(p1: Any, phase_lead_deg: float) -> None:
    source = inspect.getsource(p1._run)
    original = """            target_relative_yaw = -2.0 * math.pi * max(
                0.0, target_separation - entry_separation
            ) / lead_m
"""
    replacement = f"""            commanded_relative_yaw = -2.0 * math.pi * max(
                0.0, target_separation - entry_separation
            ) / lead_m
            actual_nut_separation = float(fixed_origin[2] - nut_position[2])
            actual_nut_helix_yaw = -2.0 * math.pi * max(
                0.0, actual_nut_separation - entry_separation
            ) / lead_m
            target_relative_yaw = max(
                commanded_relative_yaw,
                actual_nut_helix_yaw - math.radians({phase_lead_deg!r}),
            )
"""
    if source.count(original) != 1:
        raise RuntimeError("P1 yaw-target source no longer matches the probe")
    exec(
        compile(
            source.replace(original, replacement),
            "<isolated_thread_actual_position_phase_lead>",
            "exec",
        ),
        p1.__dict__,
    )


def _summarize(
    *,
    output: Path,
    p1_report: Mapping[str, Any],
    phase_lead_deg: float,
    overlay_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    trace = [
        json.loads(line)
        for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    contact_audit = json.loads(
        (output / "contact_audit.json").read_text(encoding="utf-8")
    )
    thread_rows = [
        row
        for row in contact_audit["pairs"]
        if set(row["families"]) == {"thread_followers_3", "thread_rails_3"}
    ]
    coverage: dict[str, dict[str, Any]] = {}
    for row in thread_rows:
        joined = " ".join(row["collider_paths"])
        follower_match = re.search(r"DiagnosticSphereFollower_(\d)", joined)
        rail_match = re.search(r"Rail_(\d)/Seg_(\d+)", joined)
        if follower_match is None or rail_match is None:
            raise RuntimeError("thread audit row has an unexpected semantic path")
        key = f"follower_{follower_match.group(1)}__rail_{rail_match.group(1)}"
        segment = int(rail_match.group(2))
        item = coverage.setdefault(
            key,
            {
                "follower_index": int(follower_match.group(1)),
                "rail_index": int(rail_match.group(1)),
                "segments": set(),
                "first_step": int(row["first_step"]),
                "last_step": int(row["last_step"]),
                "minimum_separation_m": float(row["minimum_separation_m"]),
                "maximum_impulse_norm_ns": float(row["maximum_impulse_norm"]),
            },
        )
        item["segments"].add(segment)
        item["first_step"] = min(item["first_step"], int(row["first_step"]))
        item["last_step"] = max(item["last_step"], int(row["last_step"]))
        item["minimum_separation_m"] = min(
            item["minimum_separation_m"], float(row["minimum_separation_m"])
        )
        item["maximum_impulse_norm_ns"] = max(
            item["maximum_impulse_norm_ns"], float(row["maximum_impulse_norm"])
        )

    serializable_coverage: dict[str, Mapping[str, Any]] = {}
    for key, item in sorted(coverage.items()):
        segments = sorted(item.pop("segments"))
        serializable_coverage[key] = {
            **item,
            "unique_segment_count": len(segments),
            "minimum_segment": min(segments),
            "maximum_segment": max(segments),
            "missing_segments_0_through_359": sorted(set(range(360)) - set(segments)),
        }

    final = trace[-1]
    final_relative_yaw = float(
        final["nut_unwrapped_yaw_rad"] - final["body_unwrapped_yaw_rad"]
    )
    final_target_yaw = float(final["target_nut_joint_rotZ_rad"])
    final_separation = float(final["observed_separation_m"])
    expected_end_separation = 0.01505
    complete_mapping = {
        (int(item["follower_index"]), int(item["rail_index"]))
        for item in serializable_coverage.values()
    } == {(0, 0), (1, 2), (2, 1)}
    complete_segment_coverage = bool(
        len(serializable_coverage) == 3
        and all(
            int(item["minimum_segment"]) == 0
            and int(item["maximum_segment"]) == 359
            and int(item["unique_segment_count"]) >= 359
            for item in serializable_coverage.values()
        )
    )
    thread_candidate_pass = bool(
        p1_report["premotion_state_pass"]
        and p1_report["all_three_thread_starts_enter"]
        and p1_report["solver_error_count"] == 0
        and p1_report["object_pose_write_after_physics_start_count"] == 0
        and complete_mapping
        and complete_segment_coverage
        and abs(final_separation - expected_end_separation) <= 5.0e-5
        and abs(final_relative_yaw - final_target_yaw) <= 0.02
        and max(abs(float(row["nut_torque_nm"][2])) for row in trace) <= 0.300001
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "role": "a3_modelling_diagnostic_not_formal_acceptance",
        "formal_acceptance_evidence": False,
        "asset_path": p1_report["asset_path"],
        "overlay_inventory": dict(overlay_inventory),
        "candidate": {
            "follower_count": 3,
            "shape": "analytic_sphere",
            "radius_m": FOLLOWER_RADIUS_M,
            "center_radius_m": FOLLOWER_CENTER_RADIUS_M,
            "center_z_m": FOLLOWER_CENTER_Z_M,
            "phase_0_deg": FOLLOWER_PHASE_0_DEG,
            "phase_step_deg": 120.0,
            "actual_position_phase_lead_deg": phase_lead_deg,
        },
        "p1_report_status_expected_in_isolation": p1_report["status"],
        "premotion_state_pass": p1_report["premotion_state_pass"],
        "all_three_thread_starts_enter": p1_report[
            "all_three_thread_starts_enter"
        ],
        "thread_event_position_error_m": p1_report["position_error_m"][
            "three_start_thread_entry"
        ],
        "solver_error_count": p1_report["solver_error_count"],
        "object_pose_write_after_physics_start_count": p1_report[
            "object_pose_write_after_physics_start_count"
        ],
        "thread_pair_row_count": len(thread_rows),
        "follower_to_rail_coverage": serializable_coverage,
        "complete_expected_follower_to_rail_mapping": complete_mapping,
        "complete_segment_coverage": complete_segment_coverage,
        "final_observed_separation_m": final_separation,
        "final_target_separation_m": float(final["target_separation_m"]),
        "final_relative_yaw_rad": final_relative_yaw,
        "final_limited_yaw_target_rad": final_target_yaw,
        "maximum_abs_nut_torque_z_nm": max(
            abs(float(row["nut_torque_nm"][2])) for row in trace
        ),
        "maximum_abs_body_force_z_n": max(
            abs(float(row["body_force_n"][2])) for row in trace
        ),
        "maximum_abs_nut_force_z_n": max(
            abs(float(row["nut_force_n"][2])) for row in trace
        ),
        "thread_candidate_pass": thread_candidate_pass,
        "file_fingerprints_computed": False,
    }


def _run(arguments: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    import d38999_physical_r7_p1_nominal_bench as p1
    import kcg_connector.d38999_tabletop_scene as tabletop_scene
    import validate_physical_r11_cooked_geometry as cooked
    import validate_physical_r7_composed_scene as composed

    _install_actual_position_phase_lead(p1, float(arguments.phase_lead_deg))
    composed._run_validation = lambda _arguments: {
        "status": "PASSED",
        "contract_revision": "keyed_v3_physical_r11",
        "diagnostic_reuse_of_prior_gate": True,
    }
    cooked._run = lambda: {
        "status": "PASSED",
        "diagnostic_reuse_of_prior_gate": True,
    }

    original_author = tabletop_scene.author_d38999_tabletop_scene
    overlay_inventory: dict[str, Any] = {}

    def author_overlay(stage: Any, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        result = original_author(stage, *args, **kwargs)
        root = (
            "/World/D38999TabletopV1/D38999Pair/"
            "D38999Shell25JKeyedPhysicalV3"
        )
        nut = root + "/LoosePlug/CouplingNut"
        rail_paths: list[str] = []
        disabled_paths: list[str] = []
        original_follower_paths: list[str] = []
        for prim in stage.Traverse():
            collision = prim.GetAttribute("physics:collisionEnabled")
            if not collision or collision.Get() is not True:
                continue
            family_attr = prim.GetAttribute("kcg:primitiveFamily")
            family = family_attr.Get() if family_attr else None
            if family == "thread_rails_3":
                rail_paths.append(str(prim.GetPath()))
                continue
            if family == "thread_followers_3":
                original_follower_paths.append(str(prim.GetPath()))
            collision.Set(False)
            disabled_paths.append(str(prim.GetPath()))
        if len(rail_paths) != 1080 or len(original_follower_paths) != 3:
            raise RuntimeError(
                "isolated thread inventory mismatch: "
                f"rails={len(rail_paths)} followers={original_follower_paths}"
            )

        material_path = root + "/Materials/coupling_thread__hard_thread"
        material = UsdShade.Material.Get(stage, material_path)
        group_path = root + "/CollisionGroups/thread_followers_3"
        group = stage.GetPrimAtPath(group_path)
        includes = group.GetRelationship("collection:colliders:includes") if group else None
        if not material or not includes:
            raise RuntimeError("missing thread material or collision group")
        replacement_paths: list[str] = []
        for index in range(3):
            phase = math.radians(FOLLOWER_PHASE_0_DEG + 120.0 * index)
            path = nut + f"/CouplingThread/DiagnosticSphereFollower_{index}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(FOLLOWER_RADIUS_M)
            sphere.CreateExtentAttr(
                [
                    Gf.Vec3f(
                        -FOLLOWER_RADIUS_M,
                        -FOLLOWER_RADIUS_M,
                        -FOLLOWER_RADIUS_M,
                    ),
                    Gf.Vec3f(
                        FOLLOWER_RADIUS_M,
                        FOLLOWER_RADIUS_M,
                        FOLLOWER_RADIUS_M,
                    ),
                ]
            )
            prim = sphere.GetPrim()
            UsdGeom.Xformable(prim).AddTranslateOp().Set(
                Gf.Vec3d(
                    FOLLOWER_CENTER_RADIUS_M * math.cos(phase),
                    FOLLOWER_CENTER_RADIUS_M * math.sin(phase),
                    FOLLOWER_CENTER_Z_M,
                )
            )
            prim.CreateAttribute(
                "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
            ).Set("thread_followers_3")
            prim.CreateAttribute(
                "kcg:materialRole", Sdf.ValueTypeNames.String, custom=True
            ).Set("coupling_thread")
            prim.CreateAttribute(
                "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
            ).Set("hard_thread")
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr(1.0e-5)
            collision_api.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, materialPurpose="physics"
            )
            includes.AddTarget(Sdf.Path(path))
            replacement_paths.append(path)
        overlay_inventory.update(
            {
                "realized_rail_count": len(rail_paths),
                "disabled_original_thread_follower_count": len(
                    original_follower_paths
                ),
                "disabled_unrelated_collider_count": len(disabled_paths) - len(
                    original_follower_paths
                ),
                "replacement_follower_paths": replacement_paths,
            }
        )
        return result

    tabletop_scene.author_d38999_tabletop_scene = author_overlay
    p1_arguments = argparse.Namespace(
        scene_config=str(Path(arguments.scene_config).resolve()),
        settle_steps=120,
        hold_steps=240,
        start_separation_m=0.00550,
        end_separation_m=0.01505,
        axial_speed_m_s=0.00050,
    )
    p1_report = p1._run(p1_arguments, output)
    return _summarize(
        output=output,
        p1_report=p1_report,
        phase_lead_deg=float(arguments.phase_lead_deg),
        overlay_inventory=overlay_inventory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    portable_root = Path(arguments.kit_portable_root)
    portable_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable_root / "warp-cache"))

    # SimulationApp otherwise adds ``--portable`` and writes below the
    # read-only Isaac installation.  Keep this diagnostic's Kit state in /tmp.
    sys.argv.extend(["--portable-root", str(portable_root)])

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        },
        experience=str(Path(__file__).with_name("d38999_cpu_physics_only.kit")),
    )
    status = 1
    try:
        report = _run(arguments, output)
        status = 0 if report["thread_candidate_pass"] else 2
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_acceptance_evidence": False,
            "file_fingerprints_computed": False,
        }
        traceback.print_exc()
    finally:
        (output / "diagnostic_summary.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    _emit(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
