#!/usr/bin/env python3

"""Probe a round detent-follower candidate against the realized r11 cam.

The r11 asset is overlaid only in memory: its three thin box followers are
disabled and replaced by three analytic round followers with the same
material.  Positive and reverse rotation use the unchanged 0.30 N*m safety
limit.  This is an A3 modelling diagnostic, not formal acceptance evidence,
and it computes no file fingerprint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Sequence


SCHEMA_VERSION = "kcg_d38999_physical_r11_round_detent_probe_v1"
GENERATOR_ID = "kcg_d38999_physical_r11_round_detent_realized_probe_v1"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step-count", type=int, default=480)
    parser.add_argument("--stiffness-n-m", type=float, default=110000.0)
    parser.add_argument("--damping-n-s-m", type=float, default=2.0)
    parser.add_argument("--angular-damping-nm-s-rad", type=float, default=0.10)
    result = parser.parse_args(argv)
    if not result.run:
        parser.error("the round-detent probe requires --run")
    if result.step_count < 240 or result.step_count > 2000:
        parser.error("step count must be in [240, 2000]")
    if not math.isfinite(result.stiffness_n_m) or result.stiffness_n_m <= 0.0:
        parser.error("stiffness must be finite and positive")
    if not math.isfinite(result.damping_n_s_m) or result.damping_n_s_m < 0.0:
        parser.error("damping must be finite and nonnegative")
    if (
        not math.isfinite(result.angular_damping_nm_s_rad)
        or result.angular_damping_nm_s_rad <= 0.0
        or result.angular_damping_nm_s_rad > 0.30
    ):
        parser.error("angular damping must be in (0, 0.30]")
    return result


def _emit(value: Any) -> None:
    os.write(
        1,
        (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    from diagnose_physical_r8_detent_lock import _run_case

    common = dict(
        disable_detent_followers=False,
        detent_follower_phase_offset_deg=-4.491137,
        detent_stiffness_n_m=float(arguments.stiffness_n_m),
        detent_damping_n_s_m=float(arguments.damping_n_s_m),
        disable_nut_body_shoulders=True,
        replace_detent_followers_with_analytic_cylinders=True,
        analytic_follower_shape="sphere",
        analytic_follower_radius_m=0.000075,
        analytic_follower_center_radius_m=0.022049,
        disable_detent_continuous_base=True,
        step_count=int(arguments.step_count),
        target_yaw_limit_rad=0.35,
        body_yaw_position_gain_nm_rad=0.8,
        nut_yaw_position_gain_nm_rad=0.8,
        angular_velocity_gain_nm_s_rad=float(
            arguments.angular_damping_nm_s_rad
        ),
        torque_component_limit_nm=0.30,
    )
    positive = _run_case(
        **common,
        target_yaw_rate_rad_s=0.412335167120566,
    )
    reverse = _run_case(
        **common,
        target_yaw_rate_rad_s=-0.412335167120566,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "role": "a3_modelling_diagnostic_not_formal_acceptance",
        "asset_revision_under_test": "keyed_v3_physical_r11",
        "candidate": {
            "shape": "analytic_sphere_follower",
            "count": 3,
            "radius_m": 0.000075,
            "center_radius_m": 0.022049,
            "base_cam_radius_m": 0.021975,
            "base_preload_m": 0.000001,
            "phase_offset_deg": -4.491137,
            "stiffness_n_m": float(arguments.stiffness_n_m),
            "damping_n_s_m": float(arguments.damping_n_s_m),
            "diagnostic_angular_damping_nm_s_rad": float(
                arguments.angular_damping_nm_s_rad
            ),
        },
        "positive_coupling": positive,
        "reverse_decoupling": reverse,
        "object_pose_write_after_physics_start_count": 0,
        "file_fingerprints_computed": False,
        "formal_acceptance_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    try:
        report = _run(arguments)
        status = 0
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "file_fingerprints_computed": False,
            "formal_acceptance_evidence": False,
        }
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    _emit(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
