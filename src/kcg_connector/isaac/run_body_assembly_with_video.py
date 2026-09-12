#!/usr/bin/env python3
"""Run the existing assembly entry point with synchronized observational video.

The same wrapper supports preflight so camera isolation can be checked before
recording motion. Control continues to receive only its existing observations.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).with_name("carts_v2")))
import run_grasp_lift as runner

invocation_arguments = list(sys.argv)
video_parser = argparse.ArgumentParser(add_help=False)
video_parser.add_argument("--assembly-video-fps", type=int, choices=(5, 10, 15, 30), default=15)
video_options, runner_arguments = video_parser.parse_known_args()
sys.argv = [sys.argv[0], *runner_arguments]


original_controller = runner._run_controller
original_execute = runner._execute
recording_state = {}


def recorded_controller(runtime, arguments, motion_plan, dynamic):
    from te_body_assembly_video import BodyAssemblyVideo

    original_stepper = runner.control.JointSignalStepper
    repository = Path(__file__).resolve().parents[3]
    output = Path(arguments.output_directory).resolve()
    # Retain video options removed by this wrapper as well as runner options.
    (output / "run_invocation.json").write_text(json.dumps({
        "executable": sys.executable, "argv": invocation_arguments,
        "working_directory": str(Path.cwd()), "simulation_only": True,
        "hardware_authorized": False,
    }, ensure_ascii=False, indent=2) + "\n")

    class RecordedStepper(original_stepper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            world, robot = kwargs["world"], kwargs["robot"]
            before_time = float(world.current_time)
            before_joint = runner._host_array(robot.get_dof_positions()).copy()
            recording_state["runtime"] = runtime
            recording_state["output"] = output
            runtime["body_assembly_video"] = BodyAssemblyVideo(
                repository, runtime, output / "video", float(dynamic["physics_dt_s"]),
                fps=video_options.assembly_video_fps)
            after_joint = runner._host_array(robot.get_dof_positions()).copy()
            isolation = dict(world_time_before_s=before_time,
                             world_time_after_s=float(world.current_time),
                             maximum_joint_change_rad=float(np.max(np.abs(after_joint-before_joint))))
            recording_state["initialization"] = isolation
            if isolation["world_time_after_s"] != before_time or isolation["maximum_joint_change_rad"] > 1e-12:
                raise RuntimeError("camera initialization changed physics time or joint state")

    runner.control.JointSignalStepper = RecordedStepper
    try:
        return original_controller(runtime, arguments, motion_plan, dynamic)
    finally:
        runner.control.JointSignalStepper = original_stepper


def recorded_execute(*args, **kwargs):
    try:
        return original_execute(*args, **kwargs)
    finally:
        runtime = recording_state.get("runtime", {})
        raw_archive=runtime.get("truth_stream")
        if raw_archive is not None and not raw_archive.closed:
            raw_archive.close()
        recorder = runtime.pop("body_assembly_video", None)
        if recorder is not None:
            video = recorder.close()
            (recording_state["output"] / "video_render_isolation.json").write_text(json.dumps({
                "initialization": recording_state.get("initialization"), "video": video,
                "source": "CURRENT_EPISODE_CAMERA_CAPTURE_AFTER_PHYSICS_STEPS",
                "physics_trajectory_comparison_required_for_initial_preflight": True,
                "video_does_not_establish_assembly_completion": True,
            }, indent=2) + "\n")


runner._run_controller = recorded_controller
runner._execute = recorded_execute

if __name__ == "__main__":
    raise SystemExit(runner.main())
