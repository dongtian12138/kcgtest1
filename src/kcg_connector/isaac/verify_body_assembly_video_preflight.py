#!/usr/bin/env python3
"""Run the existing preflight with four-camera capture to check render isolation.

This narrow development entry point leaves the ordinary runner unchanged. Its
recorded preflight must be compared with the matching no-video preflight before
the recorder is used for a complete assembly video.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).with_name("carts_v2")))
import run_grasp_lift as runner


original_run_controller = runner._run_controller


def run_recorded_preflight(runtime, arguments, motion_plan, dynamic):
    from te_body_assembly_video import BodyAssemblyVideo

    if arguments.mode != "preflight":
        raise ValueError("this recording-isolation check executes only the existing preflight")
    repository = Path(__file__).resolve().parents[3]
    output = Path(arguments.output_directory).resolve()
    original_stepper = runner.control.JointSignalStepper
    initialization = {}

    class RecordedStepper(original_stepper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            world, robot = kwargs["world"], kwargs["robot"]
            before_time = float(world.current_time)
            before_joint = runner._host_array(robot.get_dof_positions()).copy()
            runtime["body_assembly_video"] = BodyAssemblyVideo(
                repository, runtime, output / "video", float(dynamic["physics_dt_s"]))
            after_joint = runner._host_array(robot.get_dof_positions()).copy()
            initialization.update(
                world_time_before_s=before_time, world_time_after_s=float(world.current_time),
                maximum_joint_change_rad=float(np.max(np.abs(after_joint-before_joint))),
                source="JOINT_ENCODERS_AROUND_VIDEO_INITIALIZATION")
            if (initialization["world_time_after_s"] != before_time
                    or initialization["maximum_joint_change_rad"] > 1e-12):
                raise RuntimeError("camera initialization changed physics time or robot state")

    runner.control.JointSignalStepper = RecordedStepper
    try:
        return original_run_controller(runtime, arguments, motion_plan, dynamic)
    finally:
        runner.control.JointSignalStepper = original_stepper
        recorder = runtime.pop("body_assembly_video", None)
        result = recorder.close() if recorder is not None else None
        (output / "video_preflight_isolation.json").write_text(json.dumps({
            "initialization": initialization, "video": result,
            "requires_comparison_with_matching_no_video_preflight": True,
            "full_assembly_was_not_executed_by_this_preflight_check": True,
        }, indent=2) + "\n")


runner._run_controller = run_recorded_preflight

if __name__ == "__main__":
    raise SystemExit(runner.main())
