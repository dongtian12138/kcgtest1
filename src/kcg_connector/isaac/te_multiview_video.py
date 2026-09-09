#!/usr/bin/env python3
"""Synchronized four-view video capture for the TE visual grasp runtime.

The recorder is observation-only: it renders cameras after selected physics
steps and never returns image data to motion, grasp, or safety control.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Mapping

import cv2
import numpy as np


OUTPUT_RESOLUTION = (1920, 1080)
MAIN_SOURCE_RESOLUTION = (1200, 900)
SMALL_SOURCE_RESOLUTION = (640, 360)
MAIN_PANEL = (0, 0, 1440, 1080)
SMALL_PANELS = {
    "global": (1440, 0, 480, 360),
    "palm": (1440, 360, 480, 360),
    "wrist": (1440, 720, 480, 360),
}


def refresh_inspection_view(world) -> dict[str, object]:
    """Service UI/render events while keeping the explicit physics step fixed."""
    import carb
    import omni.kit.app
    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager

    before_time = float(world.current_time)
    before_step = int(world.current_time_step_index)
    timeline = omni.timeline.get_timeline_interface()
    before_timeline = float(timeline.get_current_time())
    auto_update = bool(timeline.is_auto_updating())
    settings = carb.settings.get_settings()
    play_simulations = settings.get("/app/player/playSimulations")
    if SimulationManager.is_fabric_enabled():
        from omni.physxfabric import get_physx_fabric_interface
        get_physx_fabric_interface().force_update(world.get_physics_dt(), before_time)
    try:
        timeline.set_auto_update(False)
        timeline.commit_silently()
        settings.set("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
    finally:
        settings.set("/app/player/playSimulations", play_simulations)
        timeline.set_auto_update(auto_update)
        timeline.commit_silently()
    audit = dict(physics_time_before_s=before_time, physics_time_after_s=float(world.current_time),
                 physics_step_before=before_step, physics_step_after=int(world.current_time_step_index),
                 timeline_before_s=before_timeline, timeline_after_s=float(timeline.get_current_time()))
    if audit['physics_time_after_s'] != before_time or audit['physics_step_after'] != before_step:
        raise RuntimeError("inspection refresh advanced physics: " + json.dumps(audit))
    return audit


def create_inspection_camera(world, source_camera_path: str) -> str | None:
    """Copy a framing camera once; its subsequent user edits belong to the user."""
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Sdf

    viewport = get_active_viewport()
    if viewport is None:
        return None
    path = "/World/AssemblyInspectionCamera"
    if not world.stage.GetPrimAtPath(path):
        layer = world.stage.GetRootLayer()
        if not Sdf.CopySpec(layer, Sdf.Path(source_camera_path), layer, Sdf.Path(path)):
            raise RuntimeError("could not create independent inspection camera")
    viewport.camera_path = path
    return path


def build_mask_overlay(rgb_path: Path, mask_path: Path) -> np.ndarray:
    """Return an RGB image with the measured mask visibly superimposed."""

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask_raw is None:
        raise RuntimeError("recognition overlay input image is missing")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != mask_raw.shape:
        raise RuntimeError("recognition RGB and mask dimensions differ")
    mask = mask_raw > 0
    if not np.any(mask):
        raise RuntimeError("recognition overlay mask is empty")
    result = rgb.astype(np.float32)
    tint = np.asarray((38.0, 224.0, 96.0), dtype=np.float32)
    result[mask] = 0.58 * result[mask] + 0.42 * tint
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (52, 255, 118), 3)
    ys, xs = np.nonzero(mask)
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    cv2.rectangle(result, (x0, y0), (x1, y1), (52, 255, 118), 2)
    center = (int(np.mean(xs)), int(np.mean(ys)))
    cv2.drawMarker(
        result,
        center,
        (255, 70, 70),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    return result


class MultiViewVideoRecorder:
    """Render four synchronized cameras and stream one composed H.264 video."""

    def __init__(
        self,
        *,
        rep: object,
        world: object,
        camera_paths: Mapping[str, str],
        output_path: Path,
        physics_hz: int,
        fps: int = 30,
        rt_subframes: int = 1,
        before_phase_render: Callable[[str, int, int], None] | None = None,
    ) -> None:
        required = {"main", "global", "palm", "wrist"}
        if set(camera_paths) != required:
            raise ValueError("multiview recorder requires exactly four cameras")
        self.rep = rep
        self.world = world
        self.output_path = output_path.resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite multiview video: {self.output_path}"
            )
        self.fps = int(fps)
        self.physics_hz = int(physics_hz)
        self.rt_subframes = int(rt_subframes)
        self.before_phase_render = before_phase_render
        if self.fps <= 0 or self.physics_hz <= 0:
            raise ValueError("video and physics rates must be positive")
        ratio = self.physics_hz / self.fps
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("physics rate must be an integer multiple of video fps")
        self.step_stride = int(round(ratio))
        self.resources: dict[str, tuple[object, object]] = {}
        self.frame_count = 0
        self.phase_frames: Counter[str] = Counter()
        self.last_simulation_time_s = 0.0
        self.closed = False
        self.last_frames: dict[str, np.ndarray] | None = None

        for index, name in enumerate(("main", "global", "palm", "wrist")):
            resolution = (
                MAIN_SOURCE_RESOLUTION
                if name == "main"
                else SMALL_SOURCE_RESOLUTION
            )
            render_product = rep.create.render_product(
                camera_paths[name],
                resolution,
                name=f"TEMultiviewRenderProduct{index:02d}",
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([render_product.path])
            self.resources[name] = (render_product, annotator)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for multiview video capture")
        self.ffmpeg_log_path = self.output_path.with_suffix(".ffmpeg.log")
        self.ffmpeg_log = self.ffmpeg_log_path.open("wb")
        width, height = OUTPUT_RESOLUTION
        self.encoder = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.ffmpeg_log,
        )
        if self.encoder.stdin is None:
            raise RuntimeError("ffmpeg raw-video input pipe is unavailable")
        for _ in range(3):
            self._render_views()

    def sync_after_resume(self) -> None:
        """Rebind render transforms before the first physics step after play."""
        from omni.physxfabric import get_physx_fabric_interface
        from pxr import UsdUtils

        fabric = get_physx_fabric_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(self.world.stage).ToLongInt()
        fabric.detach_stage()
        fabric.attach_stage(stage_id)

    def _render_views(self) -> dict[str, np.ndarray]:
        # Isaac Sim 6's World.render() calls update_articulations_kinematic()
        # on GPU. Do not use that physics-state operation to record images.
        # Publish native poses to Fabric, then render at the same timeline time.
        import carb
        import omni.kit.app
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager

        if SimulationManager.is_fabric_enabled():
            from omni.physxfabric import get_physx_fabric_interface
            get_physx_fabric_interface().force_update(
                self.world.get_physics_dt(), self.world.current_time)
        was_playing = bool(self.world.is_playing())
        settings = carb.settings.get_settings()
        play_simulations = settings.get("/app/player/playSimulations")
        timeline = omni.timeline.get_timeline_interface()
        auto_update = timeline.is_auto_updating()
        timeline_time = timeline.get_current_time()
        def clock_state():
            return {
                "timeline_s": float(timeline.get_current_time()),
                "physics_s": float(self.world.current_time),
                "physics_step": int(self.world.current_time_step_index),
                "timeline_playing": bool(timeline.is_playing()),
                "timeline_auto_update": bool(timeline.is_auto_updating()),
                "play_simulations": settings.get("/app/player/playSimulations"),
            }
        render_audit = {"before": clock_state()}
        try:
            timeline.set_auto_update(False)
            timeline.commit_silently()
            settings.set("/app/player/playSimulations", False)
            # Flush the existing renderer pipeline without the legacy World's
            # articulation update or an advancing timeline. Replicator's graph
            # needs normal graph evaluation for its explicit zero-time capture.
            for _ in range(3):
                omni.kit.app.get_app().update()
            render_audit["after_app_flush"] = clock_state()
            settings.set("/app/player/playSimulations", play_simulations)
            self.rep.orchestrator.step(
                rt_subframes=self.rt_subframes,
                delta_time=0.0,
                pause_timeline=not was_playing,
            )
            render_audit["after_capture"] = clock_state()
        finally:
            settings.set("/app/player/playSimulations", play_simulations)
            timeline.set_auto_update(auto_update)
            timeline.commit_silently()
            render_audit["after_restore"] = clock_state()
            self.last_render_audit = render_audit
        if abs(timeline.get_current_time() - timeline_time) > 1e-9:
            raise RuntimeError("observational video advanced the timeline: " + json.dumps(render_audit))
        frames: dict[str, np.ndarray] = {}
        for name, (_, annotator) in self.resources.items():
            rgba = np.asarray(annotator.get_data())
            if rgba.ndim != 3 or rgba.shape[2] < 3 or rgba.size == 0:
                raise RuntimeError(f"multiview camera {name} returned no RGB frame")
            frames[name] = np.array(rgba[:, :, :3], dtype=np.uint8, copy=True)
        self.last_frames = frames
        return frames

    @staticmethod
    def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        source_height, source_width = frame.shape[:2]
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        panel = np.full((height, width, 3), 17, dtype=np.uint8)
        x0 = (width - resized_width) // 2
        y0 = (height - resized_height) // 2
        panel[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
        return panel

    def _compose(
        self,
        frames: Mapping[str, np.ndarray],
        *,
        phase: str,
        simulation_time_s: float,
    ) -> np.ndarray:
        width, height = OUTPUT_RESOLUTION
        canvas = np.full((height, width, 3), 13, dtype=np.uint8)
        mx, my, mw, mh = MAIN_PANEL
        canvas[my : my + mh, mx : mx + mw] = self._fit(
            frames["main"], (mw, mh)
        )
        for name, (x, y, w, h) in SMALL_PANELS.items():
            canvas[y : y + h, x : x + w] = self._fit(frames[name], (w, h))

        return canvas

    def _write_composite(
        self,
        frames: Mapping[str, np.ndarray],
        *,
        phase: str,
        simulation_time_s: float,
    ) -> None:
        if self.closed:
            raise RuntimeError("multiview recorder is already closed")
        composite = np.ascontiguousarray(
            self._compose(
                frames,
                phase=phase,
                simulation_time_s=simulation_time_s,
            )
        )
        try:
            self.encoder.stdin.write(composite.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(
                f"ffmpeg stopped while writing {self.output_path}; "
                f"see {self.ffmpeg_log_path}"
            ) from error
        self.frame_count += 1
        self.phase_frames[phase] += 1
        self.last_simulation_time_s = float(simulation_time_s)
        if self.phase_frames[phase] == 1 or self.frame_count % self.fps == 0:
            cv2.imwrite(
                str(self.output_path.with_suffix(".preview.jpg")),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )
            print(f"VIDEO_FRAME {self.frame_count} {phase} {simulation_time_s:.4f}", flush=True)

    def capture_step(self, *, step: int, phase: str, simulation_time_s: float) -> None:
        if int(step) % self.step_stride != 0:
            return
        if self.before_phase_render is not None:
            self.before_phase_render(
                str(phase), int(self.phase_frames[str(phase)]), self.fps
            )
        self._write_composite(
            self._render_views(),
            phase=str(phase),
            simulation_time_s=float(simulation_time_s),
        )

    def write_hold(
        self,
        duration_s: float,
        *,
        phase: str,
        panel_overrides: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        count = max(1, int(round(float(duration_s) * self.fps)))
        if self.before_phase_render is not None:
            self.before_phase_render(
                str(phase), int(self.phase_frames[str(phase)]), self.fps
            )
        frames = dict(self.last_frames) if self.last_frames is not None else self._render_views()
        if panel_overrides:
            for name, frame in panel_overrides.items():
                if name not in frames:
                    raise ValueError(f"unknown video panel override: {name}")
                frames[name] = np.asarray(frame, dtype=np.uint8)
        for _ in range(count):
            self._write_composite(
                frames,
                phase=phase,
                simulation_time_s=self.last_simulation_time_s,
            )

    def close(self) -> dict[str, object]:
        if self.closed:
            raise RuntimeError("multiview recorder close called twice")
        self.closed = True
        encoding_error: str | None = None
        if self.encoder.stdin is not None:
            self.encoder.stdin.close()
        try:
            return_code = int(self.encoder.wait(timeout=120.0))
        except subprocess.TimeoutExpired:
            self.encoder.kill()
            return_code = int(self.encoder.wait())
            encoding_error = "ffmpeg did not finish within 120 seconds"
        self.ffmpeg_log.close()
        for render_product, annotator in reversed(tuple(self.resources.values())):
            try:
                annotator.detach()
            except Exception:
                pass
            try:
                render_product.destroy()
            except Exception:
                pass
        self.resources.clear()
        if return_code != 0:
            encoding_error = encoding_error or f"ffmpeg exited with {return_code}"
        if self.frame_count <= 0:
            encoding_error = encoding_error or "video contains no frames"
        if not self.output_path.is_file() or self.output_path.stat().st_size <= 0:
            encoding_error = encoding_error or "video output file is missing"
        summary = {
            "path": str(self.output_path),
            "fps": self.fps,
            "resolution_px": list(OUTPUT_RESOLUTION),
            "physics_hz": self.physics_hz,
            "physics_step_stride": self.step_stride,
            "frame_count": self.frame_count,
            "duration_s": self.frame_count / self.fps,
            "phase_frame_counts": dict(self.phase_frames),
            "ffmpeg_return_code": return_code,
            "ffmpeg_log": str(self.ffmpeg_log_path),
            "file_size_bytes": (
                self.output_path.stat().st_size
                if self.output_path.is_file()
                else 0
            ),
            "observation_only_not_returned_to_control": True,
            "error": encoding_error,
        }
        if encoding_error is not None:
            raise RuntimeError(
                f"multiview video encoding failed: {encoding_error}; "
                f"see {self.ffmpeg_log_path}"
            )
        return summary
