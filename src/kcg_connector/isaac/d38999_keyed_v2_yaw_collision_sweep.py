#!/usr/bin/env python3

"""Bounded, evidence-only yaw collision sweep for public-spec keyed-v2.

All yaw and axial-gap commands are fixed before physics starts.  Neither a
contact report nor a measured pose can change the next command.  PhysX contact
identity and rigid-body pose are captured only after each command for offline
audit of whether the five-key geometry blocks a wrong yaw before the visual
electrical-contact plane.

The output is simulation-only evidence.  It cannot authorize insertion
control, claim real-hardware clearance, or validate the unmodeled thread.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from kcg_connector.d38999_keyed_public_spec_v2 import (
    PAIR_MODEL_ID,
    PLUG_MODEL_ID,
    RECEPTACLE_MODEL_ID,
    RECOMMENDED_ASSET_NAME,
    ROOT_PRIM,
    load_keyed_public_spec_v2,
)


SCHEMA_VERSION = "kcg_d38999_keyed_v2_yaw_collision_sweep_v1"
YAW_SWEEP_DEG = (
    0.0,
    -0.02,
    0.02,
    -0.03,
    0.03,
    -0.04,
    0.04,
    -0.2,
    0.2,
    -0.333,
    0.333,
    -0.35,
    0.35,
    -0.5,
    0.5,
    180.0,
)
WRONG_YAW_AUDIT_CASES_DEG = (-0.5, 0.5, 180.0)
CORE_YAW_SWEEP_DEG = (0.0, *WRONG_YAW_AUDIT_CASES_DEG)
DEFAULT_START_GAP_M = 0.0020
DEFAULT_STOP_GAP_M = -0.0145
DEFAULT_GAP_STEP_M = 0.0001
DEFAULT_SETTLE_FRAMES = 3
MAX_GAP_COMMAND_COUNT = 500
TRIAL_SPACING_M = 0.070
CONTACT_PLANE_GAP_M = -0.012
KEYWAY_ENTRY_GAP_M = 0.0
GENERATED_ASSET_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/keyed_v2_contact_offset_r2"
) / RECOMMENDED_ASSET_NAME


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_asset_path() -> Path:
    return repository_root() / GENERATED_ASSET_RELATIVE_PATH


def default_output_dir() -> Path:
    return (
        repository_root()
        / "artifacts/kcg_connector/d38999_keyed_v2_yaw_collision_sweep_v1"
    )


def safe_new_output_dir(path: Path | str) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sweep output: {output}")
    return output


def validate_asset_path(path: Path | str) -> Path:
    asset = Path(path).expanduser().resolve()
    if asset.name != RECOMMENDED_ASSET_NAME:
        raise ValueError(
            f"yaw sweep requires the new keyed-v2 asset {RECOMMENDED_ASSET_NAME}"
        )
    if not asset.is_file():
        raise FileNotFoundError(asset)
    return asset


def build_gap_schedule(
    start_gap_m: float = DEFAULT_START_GAP_M,
    stop_gap_m: float = DEFAULT_STOP_GAP_M,
    step_m: float = DEFAULT_GAP_STEP_M,
) -> tuple[float, ...]:
    """Return a finite, strictly decreasing command schedule including ends."""

    values = tuple(float(value) for value in (start_gap_m, stop_gap_m, step_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gap schedule values must be finite")
    start, stop, step = values
    if start <= stop or step <= 0.0:
        raise ValueError("gap schedule requires start > stop and positive step")
    intervals_float = (start - stop) / step
    intervals = int(round(intervals_float))
    if not math.isclose(intervals_float, intervals, abs_tol=1.0e-9):
        raise ValueError("gap range must be an integer multiple of the step")
    if intervals + 1 > MAX_GAP_COMMAND_COUNT:
        raise ValueError("gap schedule exceeds the bounded command count")
    schedule = tuple(round(start - index * step, 12) for index in range(intervals + 1))
    if schedule[-1] != round(stop, 12):
        raise RuntimeError("gap schedule did not retain its stop bound")
    if any(first <= second for first, second in zip(schedule, schedule[1:])):
        raise RuntimeError("gap schedule must be strictly decreasing")
    return schedule


def trial_id(yaw_deg: float) -> str:
    token = f"{float(yaw_deg):+.3f}".replace("+", "p").replace("-", "m")
    return "yaw_" + token.replace(".", "p") + "deg"


def build_trials(yaws_deg: Sequence[float] = YAW_SWEEP_DEG) -> tuple[dict, ...]:
    return tuple(
        {
            "trial_index": index,
            "trial_id": trial_id(yaw_deg),
            "yaw_deg": yaw_deg,
            "x_offset_m": (index % 4) * TRIAL_SPACING_M,
            "y_offset_m": (index // 4) * TRIAL_SPACING_M,
        }
        for index, yaw_deg in enumerate(yaws_deg)
    )


def classify_pair_contact(
    paths: Sequence[str], *, body_path: str, fixed_path: str
) -> str | None:
    """Classify paths for offline audit; never used to choose motion."""

    normalized = tuple(str(path) for path in paths)
    has_body = any(
        path == body_path or path.startswith(body_path + "/")
        for path in normalized
    )
    has_fixed = any(
        path == fixed_path or path.startswith(fixed_path + "/")
        for path in normalized
    )
    if not (has_body and has_fixed):
        return None
    has_plug_key = any("/CollisionKeys/" in path for path in normalized)
    has_fixed_keyway = any("/CollisionKeyways/" in path for path in normalized)
    if has_plug_key or has_fixed_keyway:
        return "polarization_key_or_keyway"
    return "pair_other_collision"


def _first_contact_sample(samples: Sequence[Mapping], kind: str | None):
    for index, sample in enumerate(samples):
        contacts = tuple(sample.get("contacts", ()))
        matching = (
            contacts
            if kind is None
            else tuple(item for item in contacts if item.get("kind") == kind)
        )
        if matching:
            previous_gap = (
                None
                if index == 0
                else float(samples[index - 1]["commanded_gap_m"])
            )
            return {
                "index": index,
                "commanded_gap_m": float(sample["commanded_gap_m"]),
                "previous_commanded_gap_m": previous_gap,
                "measured_body_tip_z_m": sample.get("measured_body_tip_z_m"),
                "contacts": list(matching),
            }
    return None


def summarize_trial(
    trial: Mapping,
    samples: Sequence[Mapping],
    *,
    contact_plane_gap_m: float = CONTACT_PLANE_GAP_M,
) -> dict:
    """Summarize discrete contact-onset brackets without claiming a pass."""

    first_pair = _first_contact_sample(samples, None)
    first_polarization = _first_contact_sample(
        samples, "polarization_key_or_keyway"
    )
    blocked_before_visual_contact = bool(
        first_polarization is not None
        and first_polarization["commanded_gap_m"] > contact_plane_gap_m
    )
    commanded_through_contact_plane = any(
        float(sample["commanded_gap_m"]) <= contact_plane_gap_m
        for sample in samples
    )
    first_contact_is_polarization = bool(
        first_pair is not None
        and first_polarization is not None
        and first_pair["index"] == first_polarization["index"]
    )
    entry_onset_bracketed = bool(
        first_polarization is not None
        and first_polarization["previous_commanded_gap_m"] is not None
        and first_polarization["commanded_gap_m"] <= KEYWAY_ENTRY_GAP_M
        < first_polarization["previous_commanded_gap_m"]
    )
    pair_clear_to_visual_contact_plane = bool(
        commanded_through_contact_plane
        and (
            first_pair is None
            or first_pair["commanded_gap_m"] <= contact_plane_gap_m
        )
    )
    polarization_clear_to_contact_plane = bool(
        commanded_through_contact_plane
        and not blocked_before_visual_contact
    )
    if (
        blocked_before_visual_contact
        and first_contact_is_polarization
        and entry_onset_bracketed
    ):
        evidence = (
            "POLARIZATION_CONTACT_AT_KEYWAY_ENTRY_BEFORE_"
            "VISUAL_ELECTRICAL_CONTACT_PLANE"
        )
    elif blocked_before_visual_contact:
        evidence = (
            "POLARIZATION_CONTACT_BEFORE_VISUAL_ELECTRICAL_CONTACT_PLANE_"
            "WITHOUT_ENTRY_ONSET_PROOF"
        )
    elif first_polarization is not None:
        evidence = "POLARIZATION_CONTACT_NOT_BEFORE_VISUAL_ELECTRICAL_PLANE"
    elif first_pair is not None:
        evidence = "CONTACT_OBSERVED_WITHOUT_POLARIZATION_ATTRIBUTION"
    else:
        evidence = "NO_CONTACT_OBSERVED_WITHIN_BOUNDED_SWEEP"
    return {
        "trial_id": str(trial["trial_id"]),
        "yaw_deg": float(trial["yaw_deg"]),
        "sample_count": len(samples),
        "contact_plane_gap_m": float(contact_plane_gap_m),
        "first_pair_contact": first_pair,
        "first_polarization_contact": first_polarization,
        "keyway_entry_gap_m": KEYWAY_ENTRY_GAP_M,
        "polarization_onset_gap_bracket_m": (
            None
            if first_polarization is None
            else [
                first_polarization["commanded_gap_m"],
                first_polarization["previous_commanded_gap_m"],
            ]
        ),
        "polarization_onset_brackets_keyway_entry": entry_onset_bracketed,
        "first_pair_contact_is_polarization_contact": (
            first_contact_is_polarization
        ),
        "minimum_visual_electrical_plane_margin_m": (
            None
            if first_polarization is None
            else float(
                first_polarization["commanded_gap_m"] - contact_plane_gap_m
            )
        ),
        "commanded_through_contact_plane": commanded_through_contact_plane,
        "pair_clear_to_visual_electrical_contact_plane": (
            pair_clear_to_visual_contact_plane
        ),
        "polarization_blocked_before_contact_plane": (
            blocked_before_visual_contact
        ),
        "polarization_clear_to_contact_plane": polarization_clear_to_contact_plane,
        "evidence_verdict": evidence,
        "formal_acceptance": "NOT_EVALUATED_EVIDENCE_ONLY",
        "control_promotion_allowed": False,
    }


def summarize_sweep(trial_summaries: Sequence[Mapping]) -> dict:
    by_yaw = {
        round(float(summary["yaw_deg"]), 9): summary
        for summary in trial_summaries
    }
    cases = []
    for yaw_deg in WRONG_YAW_AUDIT_CASES_DEG:
        summary = by_yaw.get(round(yaw_deg, 9))
        cases.append(
            {
                "yaw_deg": yaw_deg,
                "evidence_present": summary is not None,
                "polarization_blocked_before_contact_plane": bool(
                    summary is not None
                    and summary.get(
                        "polarization_blocked_before_contact_plane", False
                    )
                    and summary.get(
                        "first_pair_contact_is_polarization_contact", False
                    )
                    and summary.get(
                        "polarization_onset_brackets_keyway_entry", False
                    )
                ),
            }
        )
    complete = all(
        item["evidence_present"]
        and item["polarization_blocked_before_contact_plane"]
        for item in cases
    )
    correct_yaw = by_yaw.get(0.0)
    correct_yaw_clear = bool(
        correct_yaw is not None
        and correct_yaw.get(
            "pair_clear_to_visual_electrical_contact_plane", False
        )
    )
    return {
        "correct_n_yaw_case": {
            "yaw_deg": 0.0,
            "evidence_present": correct_yaw is not None,
            "commanded_through_contact_plane": bool(
                correct_yaw is not None
                and correct_yaw.get("commanded_through_contact_plane", False)
            ),
            "polarization_clear_to_contact_plane": correct_yaw_clear,
            "pair_clear_to_visual_electrical_contact_plane": (
                correct_yaw_clear
            ),
        },
        "required_wrong_yaw_cases": cases,
        "wrong_yaw_key_block_evidence_complete": complete,
        "correct_and_wrong_yaw_evidence_complete": bool(
            correct_yaw_clear and complete
        ),
        "result_kind": "SIMULATION_ONLY_OFFLINE_COLLISION_EVIDENCE",
        "formal_acceptance": "NOT_EVALUATED_EVIDENCE_ONLY",
        "simulation_insertion_control_authorized": False,
        "robot_control_authorized": False,
        "hardware_control_authorized": False,
        "coupling_ring_initial_engagement_sequence_evaluated": False,
        "electrical_contact_sequence_physics_evaluated": False,
    }


def _arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the independent keyed-v2 yaw collision evidence sweep"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="run only 0, +/-0.5 and 180 degree preflight cases",
    )
    parser.add_argument("--asset", default=str(default_asset_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--start-gap-mm", type=float, default=2.0)
    parser.add_argument("--stop-gap-mm", type=float, default=-14.5)
    parser.add_argument("--gap-step-mm", type=float, default=0.1)
    parser.add_argument(
        "--settle-frames", type=int, default=DEFAULT_SETTLE_FRAMES
    )
    arguments = parser.parse_args(argv)
    if not arguments.run:
        parser.error("yaw collision sweep requires explicit --run")
    if not 1 <= arguments.settle_frames <= 20:
        parser.error("--settle-frames must stay within [1, 20]")
    return arguments


def main(argv=None):
    arguments = _arguments(argv)
    model = load_keyed_public_spec_v2()
    asset_path = validate_asset_path(arguments.asset)
    output_dir = safe_new_output_dir(arguments.output_dir)
    gap_step_m = arguments.gap_step_mm * 1.0e-3
    gaps = build_gap_schedule(
        arguments.start_gap_mm * 1.0e-3,
        arguments.stop_gap_mm * 1.0e-3,
        gap_step_m,
    )
    selected_yaws = CORE_YAW_SWEEP_DEG if arguments.core_only else YAW_SWEEP_DEG
    trials = build_trials(selected_yaws)
    output_dir.mkdir(parents=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "asset_path": str(asset_path),
        "asset_basename": asset_path.name,
        "pair_model_id": PAIR_MODEL_ID,
        "plug_model_id": PLUG_MODEL_ID,
        "receptacle_model_id": RECEPTACLE_MODEL_ID,
        "mode": "SIMULATION_ONLY_OFFLINE_COLLISION_EVIDENCE",
        "control_authorized": False,
        "thread_collision_mode": "unmodeled",
        "contact_collision_mode": "visual_only",
        "keyway_entry_gap_m": KEYWAY_ENTRY_GAP_M,
        "coupling_ring_initial_engagement_gap_m": None,
        "coupling_ring_initial_engagement_sequence_evaluated": False,
        "electrical_contact_sequence_physics_evaluated": False,
        "command_policy": "FIXED_OPEN_LOOP_YAW_AND_GAP_GRID",
        "control_reads_contact_report": False,
        "control_reads_pose_truth": False,
        "posthoc_audit_reads_contact_report": True,
        "posthoc_audit_reads_pose_truth": True,
        "contact_or_pose_changes_next_command": False,
        "yaw_sweep_deg": list(selected_yaws),
        "core_only": bool(arguments.core_only),
        "gap_schedule": {
            "start_m": gaps[0],
            "stop_m": gaps[-1],
            "step_m": gap_step_m,
            "command_count": len(gaps),
        },
        "settle_frames_per_command": arguments.settle_frames,
        "visual_electrical_contact_plane_gap_m": CONTACT_PLANE_GAP_M,
        "run_completed": False,
    }

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    run_completed = False
    raw_samples = []
    try:
        import numpy as np

        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import get_current_stage
        from omni.physx import get_physx_simulation_interface
        from pxr import (
            Gf,
            PhysicsSchemaTools,
            PhysxSchema,
            UsdGeom,
            UsdPhysics,
        )

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
        UsdGeom.Xform.Define(stage, "/World/KeyedV2YawSweep")

        runtimes = []
        for trial in trials:
            trial_root = (
                f"/World/KeyedV2YawSweep/Trial_{trial['trial_index']:03d}"
            )
            pair = UsdGeom.Xform.Define(stage, trial_root)
            pair.GetPrim().GetReferences().AddReference(
                str(asset_path), ROOT_PRIM
            )
            UsdGeom.Xformable(pair).AddTranslateOp().Set(
                Gf.Vec3d(trial["x_offset_m"], trial["y_offset_m"], 0.0)
            )
            expected_metadata = {
                "kcg:pairModelId": PAIR_MODEL_ID,
                "kcg:loosePlugModelId": PLUG_MODEL_ID,
                "kcg:fixedReceptacleModelId": RECEPTACLE_MODEL_ID,
                "kcg:threadCollisionMode": "unmodeled",
                "kcg:contactCollisionMode": "visual_only",
            }
            for key, expected in expected_metadata.items():
                if pair.GetPrim().GetCustomDataByKey(key) != expected:
                    raise RuntimeError(
                        f"keyed-v2 asset metadata mismatch at {key}"
                    )

            body_path = trial_root + "/LoosePlug/BodyAssembly"
            nut_path = trial_root + "/LoosePlug/CouplingNut"
            joint_path = trial_root + "/LoosePlug/CouplingNutJoint"
            fixed_path = trial_root + "/FixedReceptacle"
            body_prim = stage.GetPrimAtPath(body_path)
            nut_prim = stage.GetPrimAtPath(nut_path)
            joint_prim = stage.GetPrimAtPath(joint_path)
            fixed_prim = stage.GetPrimAtPath(fixed_path)
            if not all(
                prim.IsValid()
                for prim in (body_prim, nut_prim, joint_prim, fixed_prim)
            ):
                raise RuntimeError("keyed-v2 asset is missing required suffixes")
            # The unmodeled thread/nut must not contaminate a polarization-key
            # collision audit.  The body remains dynamic so contact with the
            # static receptacle is resolved and reported by PhysX.
            nut_prim.SetActive(False)
            joint_prim.SetActive(False)
            UsdPhysics.RigidBodyAPI.Apply(
                body_prim
            ).CreateKinematicEnabledAttr().Set(False)
            PhysxSchema.PhysxContactReportAPI.Apply(
                body_prim
            ).CreateThresholdAttr().Set(0.0)
            runtimes.append(
                {
                    **trial,
                    "root_path": trial_root,
                    "body_path": body_path,
                    "fixed_path": fixed_path,
                    "body": world.scene.add(
                        SingleRigidPrim(
                            prim_path=body_path,
                            name=f"keyed_body_{trial['trial_index']:03d}",
                        )
                    ),
                    "samples": [],
                }
            )

        world.reset()
        world.get_physics_context().set_gravity(0.0)

        def capture_contacts_for_offline_audit():
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            grouped = {item["trial_index"]: [] for item in runtimes}
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                for runtime in runtimes:
                    kind = classify_pair_contact(
                        paths,
                        body_path=runtime["body_path"],
                        fixed_path=runtime["fixed_path"],
                    )
                    if kind is None:
                        continue
                    start = int(header.contact_data_offset)
                    stop = start + int(header.num_contact_data)
                    separations = [
                        float(contacts[index].separation)
                        for index in range(start, stop)
                    ]
                    grouped[runtime["trial_index"]].append(
                        {
                            "kind": kind,
                            "paths": list(paths),
                            "contact_record_count": int(
                                header.num_contact_data
                            ),
                            "minimum_separation_m": (
                                min(separations) if separations else None
                            ),
                        }
                    )
                    break
            return grouped

        # This schedule is immutable.  Offline observations below are appended
        # to evidence only and are never inspected to select or stop commands.
        command_schedule = tuple(gaps)
        for commanded_gap_m in command_schedule:
            for runtime in runtimes:
                half_yaw = 0.5 * math.radians(runtime["yaw_deg"])
                orientation = np.asarray(
                    (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
                    dtype=np.float64,
                )
                position = np.asarray(
                    (
                        runtime["x_offset_m"],
                        runtime["y_offset_m"],
                        -commanded_gap_m,
                    ),
                    dtype=np.float64,
                )
                runtime["body"].set_world_pose(
                    position=position, orientation=orientation
                )
                runtime["body"].set_linear_velocity((0.0, 0.0, 0.0))
                runtime["body"].set_angular_velocity((0.0, 0.0, 0.0))
            contacts_by_trial = {
                runtime["trial_index"]: [] for runtime in runtimes
            }
            for _ in range(arguments.settle_frames):
                world.step(render=arguments.gui)
                frame_contacts = capture_contacts_for_offline_audit()
                for trial_index, frame_items in frame_contacts.items():
                    contacts_by_trial[trial_index].extend(frame_items)
            for runtime in runtimes:
                measured_position, _ = runtime["body"].get_world_pose()
                sample = {
                    "trial_index": runtime["trial_index"],
                    "trial_id": runtime["trial_id"],
                    "yaw_deg": runtime["yaw_deg"],
                    "commanded_gap_m": commanded_gap_m,
                    "measured_body_tip_z_m": float(measured_position[2]),
                    "contacts": contacts_by_trial[runtime["trial_index"]],
                    "truth_scope": "POSTHOC_AUDIT_ONLY",
                }
                runtime["samples"].append(sample)
                raw_samples.append(sample)

        summaries = tuple(
            summarize_trial(runtime, runtime["samples"])
            for runtime in runtimes
        )
        report["trial_summaries"] = list(summaries)
        report["sweep_summary"] = summarize_sweep(summaries)
        report["run_completed"] = True
        report["formal_acceptance"] = "NOT_EVALUATED_EVIDENCE_ONLY"
        report["control_promotion_allowed"] = False
        report["body_motion_mode"] = (
            "DYNAMIC_BODY_POSE_RESET_ON_PRECOMMITTED_GRID"
        )
        report["coupling_nut_and_joint_isolated"] = True

        (output_dir / "samples.jsonl").write_text(
            "".join(
                json.dumps(sample, sort_keys=True) + "\n"
                for sample in raw_samples
            ),
            encoding="utf-8",
        )
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("D38999 KEYED-V2 YAW COLLISION SWEEP EVIDENCE WRITTEN")
        print(f"  output: {output_dir}")
        print("  use: simulation-only offline audit; no control authorization")
        print("  thread: unmodeled")
        run_completed = True
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        traceback.print_exc()
        print("D38999 KEYED-V2 YAW COLLISION SWEEP FAILED", flush=True)
    finally:
        if not run_completed:
            try:
                (output_dir / "report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except BaseException:
                traceback.print_exc()
        simulation_app.close(exit_code=0 if run_completed else 1)


if __name__ == "__main__":
    main()
