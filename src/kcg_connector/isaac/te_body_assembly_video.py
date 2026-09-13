"""Connect the existing four-camera recorder to the body-assembly runtime."""

from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path

import numpy as np
import yaml


class BodyAssemblyVideo:
    """Render from fixed calibration and encoder FK; never feed video to control."""

    def __init__(self, repository, runtime, output, physics_dt_s, *, fps=15):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom
        from te_body_socket_observation import GLOBAL_CAMERA_CONFIG, hand_camera_mount
        from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target
        from te_multiview_video import (MultiViewVideoRecorder, MAIN_SOURCE_RESOLUTION, SMALL_SOURCE_RESOLUTION,
                                       create_inspection_camera, refresh_inspection_view)

        self.runtime = runtime
        self.capture_wall_s = 0.0
        self.maximum_render_state_deltas = {}
        self.world = runtime["world"]
        self.stage = omni.usd.get_context().get_stage()
        self.Gf, self.UsdGeom = Gf, UsdGeom
        self.author_camera = _author_camera
        self.eye_target = _camera_cv_pose_from_eye_target
        self.main_resolution = MAIN_SOURCE_RESOLUTION
        self.small_resolution = SMALL_SOURCE_RESOLUTION
        self.physics_dt_s = float(physics_dt_s)
        self.fps = int(fps)
        self.stride = round(1.0 / self.physics_dt_s / self.fps)
        self.output = Path(output)
        self.frozen_main_target = None
        self.paths = {name: "/World/BodyAssemblyVideo/" + name for name in ("main", "global", "palm", "wrist")}
        self.mounts = {name: hand_camera_mount(repository, name) for name in ("palm", "wrist")}
        global_config = yaml.safe_load((Path(repository) / GLOBAL_CAMERA_CONFIG).read_text())["camera"]
        global_pose = self.eye_target(global_config["eye_world_m"], global_config["target_world_m"])
        self.author_camera(self.stage, self.paths["global"], global_pose,
            resolution=SMALL_SOURCE_RESOLUTION, focal_length_mm=global_config["focal_length_mm"],
            horizontal_aperture_mm=global_config["horizontal_aperture_mm"],
            clipping_range_m=tuple(global_config["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom)
        self._set_poses("initialization", np.zeros(11))
        self.recorder = MultiViewVideoRecorder(rep=rep, world=self.world, camera_paths=self.paths,
            output_path=self.output / "assembly_four_view.mp4", physics_hz=round(1.0/self.physics_dt_s), fps=self.fps)
        self.inspection_camera = (create_inspection_camera(self.world, self.paths['global'])
            if runtime.get('inspection_ui_enabled',True) else None)
        self.refresh_inspection_view = refresh_inspection_view
        self.next_ui_refresh_wall_s = perf_counter()
        self.last_ui_refresh_wall_s = None
        self.ui_refresh_count = 0
        self.ui_refresh_wall_s = 0.0
        self.maximum_ui_refresh_gap_s = 0.0
        self.ui_audit = (self.output / "inspection_refresh_audit.jsonl").open("x", buffering=1)
        self.timeline = (self.output / "assembly_four_view_frames.jsonl").open("x", encoding="utf-8", buffering=1)
        self.original_capture = runtime["auditor"].capture
        runtime["auditor"].capture = self.capture

    def freeze_main_target(self, point_from_current_vision):
        self.frozen_main_target = np.asarray(point_from_current_vision, dtype=np.float64).reshape(3).copy()

    def _set_poses(self, phase, positions):
        hand = np.asarray(self.runtime["inputs"].robot_model.forward_kinematics(
            tuple(positions), enforce_limits=False)["handbase_link"])
        for name, mount in self.mounts.items():
            self.author_camera(self.stage, self.paths[name], hand @ np.asarray(mount["hand_from_camera_cv"]),
                resolution=self.small_resolution, focal_length_mm=mount["focal_length_mm"],
                horizontal_aperture_mm=mount["horizontal_aperture_mm"],
                clipping_range_m=tuple(mount["clipping_range_m"]), Gf=self.Gf, UsdGeom=self.UsdGeom)
        close_view = phase.startswith(("parallel_contact", "finger_", "preload", "lift", "hold", "key_probe"))
        if self.frozen_main_target is not None:
            target = self.frozen_main_target
            eye = target + np.asarray([.28, -.22, .20])
        elif close_view:
            # Only a framing estimate from robot geometry, not an object pose
            # observation. It remains useful through the small regrasp offset.
            target = (hand @ np.asarray([-.00973253, .00561908, .45243714, 1.0]))[:3]
            eye = target + np.asarray([.28, -.22, .20])
        else:
            target = np.asarray([.45, -.02, .45])
            eye = np.asarray([1.15, -.9, .8])
        progress=self.runtime.get('last_nut_progress_observation')
        if phase=='nut_index_free_open_hold' and progress is not None:
            # A side close-up uses the current visual estimate, never truth or
            # an assembly-status-dependent change to the connector's red band.
            target=np.asarray(progress['observation']['world_from_plug_five_dof'])[:3,3]
            eye=target+np.asarray([.10,-.08,.015])
        self.author_camera(self.stage, self.paths["main"], self.eye_target(eye, target),
            resolution=self.main_resolution, focal_length_mm=24.0, horizontal_aperture_mm=36.0,
            clipping_range_m=(.02, 10.0), Gf=self.Gf, UsdGeom=self.UsdGeom)

    def capture(self, **kwargs):
        self.original_capture(**kwargs)
        step = int(kwargs["step"])
        wall_now = perf_counter()
        if self.inspection_camera is not None and wall_now >= self.next_ui_refresh_wall_s:
            native_before_ui = self._native_state_for_recording()
            audit = self.refresh_inspection_view(self.world)
            native_after_ui = self._native_state_for_recording()
            audit.update(step=step, phase=str(kwargs['phase']),
                         native_state_deltas_are_record_only_not_controller_inputs=True,
                         native_state_max_abs_deltas={
                             name: float(np.max(np.abs(native_after_ui[name]-value)))
                             for name, value in native_before_ui.items()})
            self.ui_audit.write(json.dumps(audit)+'\n')
            if self.last_ui_refresh_wall_s is not None:
                self.maximum_ui_refresh_gap_s = max(self.maximum_ui_refresh_gap_s,
                                                   wall_now-self.last_ui_refresh_wall_s)
            self.last_ui_refresh_wall_s = wall_now
            self.ui_refresh_count += 1
            after_ui = perf_counter()
            self.ui_refresh_wall_s += after_ui-wall_now
            self.next_ui_refresh_wall_s = after_ui+0.1
        if step % self.stride:
            return
        started = perf_counter()
        phase = str(kwargs["phase"])
        native_before = self._native_state_for_recording()
        self._set_poses(phase, np.asarray(kwargs["active_positions"]))
        before = float(self.world.current_time)
        index_before = int(self.world.current_time_step_index)
        render_error = None
        try:
            self.recorder.capture_step(step=step, phase=phase, simulation_time_s=(step+1)*self.physics_dt_s)
        except Exception as error:
            render_error = error
        after = float(self.world.current_time)
        native_after = self._native_state_for_recording()
        deltas = {name: float(np.max(np.abs(native_after[name] - value)))
                  for name, value in native_before.items()}
        for name, value in deltas.items():
            self.maximum_render_state_deltas[name] = max(
                self.maximum_render_state_deltas.get(name, 0.0), value)
        self.timeline.write(json.dumps({"frame": self.recorder.frame_count-1 if render_error is None else None, "step": step,
            "phase": phase, "simulation_time_s": (step+1)*self.physics_dt_s,
            "world_time_before_render_s": before, "world_time_after_render_s": after,
            "world_step_before_render": index_before,
            "world_step_after_render": int(self.world.current_time_step_index),
            "render_error": str(render_error) if render_error else None,
            "render_clock_audit": getattr(self.recorder, "last_render_audit", None),
            "render_native_state_max_abs_deltas": deltas,
            "native_state_deltas_are_record_only_not_controller_inputs": True}) + "\n")
        self.capture_wall_s += perf_counter() - started
        if abs(after - before) > 1e-9 or int(self.world.current_time_step_index) != index_before:
            raise RuntimeError("video rendering advanced physics time or step index") from render_error
        if render_error is not None:
            raise render_error

    def _native_state_for_recording(self):
        """Read the same native APIs on both sides of rendering; never control."""
        def host(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            elif hasattr(value, "numpy"):
                value = value.numpy()
            return np.asarray(value, dtype=np.float64).copy()

        robot = self.runtime["robot_data"][0]
        parts = self.runtime["object_parts"]
        poses = [part.get_world_pose() for part in parts]
        state = {
            "robot_joint_position_rad": host(robot.get_dof_positions(indices=0)),
            "robot_joint_velocity_rad_s": host(robot.get_dof_velocities(indices=0)),
            "robot_projected_joint_force_nm": host(robot.get_dof_projected_joint_forces(indices=0)),
            "object_position_m": np.asarray([host(pose[0]) for pose in poses]),
            "object_quaternion_wxyz": np.asarray([host(pose[1]) for pose in poses]),
            "object_linear_velocity_m_s": np.asarray([host(part.get_linear_velocity()) for part in parts]),
            "object_angular_velocity_rad_s": np.asarray([host(part.get_angular_velocity()) for part in parts]),
        }
        ft = self.runtime.get("nail_body_ft_auditor")
        if ft is not None:
            state["raw_wrist_wrench_n_nm"] = host(
                ft.ft_articulation.get_measured_joint_forces())[ft.reaction_row]
        return state

    def close(self):
        self.runtime["auditor"].capture = self.original_capture
        self.timeline.close()
        self.ui_audit.close()
        result = self.recorder.close()
        result["capture_and_encode_wall_s"] = self.capture_wall_s
        result['inspection_view'] = dict(camera_path=self.inspection_camera,
            camera_pose_rewritten_during_motion=False, refresh_count=self.ui_refresh_count,
            refresh_wall_s=self.ui_refresh_wall_s, maximum_refresh_gap_s=self.maximum_ui_refresh_gap_s,
            wall_seconds_between_refresh_attempts=0.1,
            note='Responsiveness during explicit motion; blocking planning/perception calls are not yet serviced here.')
        result["maximum_render_native_state_deltas"] = self.maximum_render_state_deltas
        result["render_state_audit_role"] = "POSTRUN_ONLY; NO_STATE_VALUES_RETURNED_TO_CONTROL"
        result.update(camera_pose_inputs="FIXED_RIG_CALIBRATION_AND_JOINT_ENCODER_FK",
                      simulator_object_or_contact_truth_used_for_camera_poses=False,
                      physics_time_invariance_checked_each_frame=True)
        (self.output / "assembly_four_view_video.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
