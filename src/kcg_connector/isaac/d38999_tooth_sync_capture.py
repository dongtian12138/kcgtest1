#!/usr/bin/env python3

"""Hash-bound, fixed four-camera tooth capture for the prepared twist probe.

The class is deliberately injected into ``d38999_nut_regrasp_smoke`` only by
an explicit CLI flag.  It owns presentation resources, never steps physics and
never writes an object, robot or constraint pose.  The caller invokes
``maybe_capture`` after a completed 240 Hz physics step; a 30 Hz schedule then
records the nearest (exactly observed) global step in ``video_frame_sync.csv``.
Four fixed rear/front oblique views are read for each sampled physics state.
No physics step or scene-pose write occurs between those reads.  The front
pair is important evidence coverage: a tooth hidden on the far side of the
nut from both rear cameras must not silently disappear from the audit.

Isaac and Replicator bindings are injected so module import and pure tests do
not start Kit or acquire GPU resources.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time


SCHEMA_VERSION = "kcg_d38999_tooth_sync_capture_v3"
DEFAULT_CAPTURE_RATE_HZ = 30
DEFAULT_RESOLUTION = (960, 720)
DEFAULT_CAMERA_EYE_M = (0.30, -0.46, 0.39)
DEFAULT_CAMERA_TARGET_M = (0.55, 0.185, 0.265)
VIEW_IDS = (
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
)
CAPTURE_PHASES = (
    "nut_only_final_hold",
    "q7_twist_probe_motion",
    "q7_twist_probe_hold",
)
SYNC_FIELDS = (
    "frame_index",
    "sample_index",
    "view_id",
    "global_step",
    "phase",
    "phase_step",
    "simulation_time_s",
    "timestamp_s",
    "rgb_filename",
)


def build_camera_rig_contract(resolution, left_eye, target):
    """Build fixed equal-distance rear/front, left/right oblique views."""

    resolution = tuple(int(value) for value in resolution)
    left_eye = tuple(float(value) for value in left_eye)
    target = tuple(float(value) for value in target)
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("capture resolution must be positive width/height")
    if len(left_eye) != 3 or len(target) != 3:
        raise ValueError("capture camera vectors must contain three values")
    if not all(math.isfinite(value) for value in (*left_eye, *target)):
        raise ValueError("capture camera vectors must be finite")
    rear_right_eye = (
        2.0 * target[0] - left_eye[0],
        left_eye[1],
        left_eye[2],
    )
    front_left_eye = (
        left_eye[0],
        2.0 * target[1] - left_eye[1],
        left_eye[2],
    )
    front_right_eye = (
        rear_right_eye[0],
        front_left_eye[1],
        left_eye[2],
    )
    definitions = (
        ("rear_left", "RearLeft", left_eye),
        ("rear_right", "RearRight", rear_right_eye),
        ("front_left", "FrontLeft", front_left_eye),
        ("front_right", "FrontRight", front_right_eye),
    )
    views = []
    for view_id, suffix, eye in definitions:
        views.append(
            {
                "eye_m": list(eye),
                "focal_length_mm": 50.0,
                "prim_path": (
                    "/World/D38999NutRegrasp/"
                    f"ToothSyncCamera{suffix}"
                ),
                "fixed_oblique_before_play": True,
                "render_product_name": (
                    f"D38999ToothSyncRenderProduct{suffix}"
                ),
                "resolution": list(resolution),
                "target_m": list(target),
                "view_id": view_id,
            }
        )
    return {
        "same_completed_physics_step": True,
        "selection_contract": {
            "ab_requires_same_view_per_transition": True,
            "minimum_adjacent_pitch_px": 12.0,
            "minimum_adjacent_pairs": 3,
            "minimum_visible_teeth": 4,
            "policy": "first_valid_view_in_manifest_priority",
        },
        "view_priority": list(VIEW_IDS),
        "views": views,
    }


def sha256_file(path):
    """Return the SHA-256 for a capture artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Lock the bytes that Python actually imported.  Constructor/finalize checks
# below reject the run if the source file changes at any later boundary.
_MODULE_SOURCE_PATH = Path(__file__).resolve()
_MODULE_SHA256_AT_IMPORT = sha256_file(_MODULE_SOURCE_PATH)


def prepare_empty_output_directory(path):
    """Create an empty evidence directory or reject stale mixed-run files."""

    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"tooth sync capture output is not empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_sampling_rates(physics_rate_hz, capture_rate_hz):
    """Require a deterministic integer-step render schedule."""

    if isinstance(physics_rate_hz, bool) or isinstance(capture_rate_hz, bool):
        raise ValueError("capture rates must be integers")
    physics = int(physics_rate_hz)
    capture = int(capture_rate_hz)
    if physics != physics_rate_hz or capture != capture_rate_hz:
        raise ValueError("capture rates must be integers")
    if capture <= 0 or physics <= 0 or capture > physics:
        raise ValueError("capture rates must satisfy 0 < capture <= physics")
    if physics % capture != 0:
        raise ValueError("physics rate must be divisible by capture rate")
    return {
        "capture_rate_hz": capture,
        "physics_rate_hz": physics,
        "physics_steps_per_frame": physics // capture,
        "sampling_kind": "fixed_integer_physics_step_decimation",
    }


def should_capture_phase_step(
    phase,
    phase_step,
    *,
    physics_steps_per_frame,
    capture_phases=CAPTURE_PHASES,
):
    """Return true at phase-local step 1 and each fixed decimation boundary."""

    if phase not in capture_phases:
        return False
    if isinstance(phase_step, bool) or not isinstance(phase_step, int):
        raise ValueError("phase_step must be an integer")
    if phase_step <= 0:
        raise ValueError("phase_step must be positive")
    if physics_steps_per_frame <= 0:
        raise ValueError("physics_steps_per_frame must be positive")
    return phase_step == 1 or phase_step % physics_steps_per_frame == 0


def expected_sampled_phase_steps(total_steps, physics_steps_per_frame):
    """Return the complete phase-local schedule, including phase step one."""

    if total_steps <= 0:
        raise ValueError("total phase steps must be positive")
    multiples = list(
        range(
            physics_steps_per_frame,
            total_steps + 1,
            physics_steps_per_frame,
        )
    )
    return multiples if multiples and multiples[0] == 1 else [1, *multiples]


def validate_sync_rows(
    rows,
    sampling,
    phase_step_totals=None,
    view_ids=VIEW_IDS,
):
    """Fail closed unless every logical sample contains every fixed view."""

    previous_frame = -1
    previous_sample_step = -1
    seen_phases = set()
    samples = []
    view_ids = tuple(view_ids)
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError("view IDs must be unique and non-empty")
    if not rows:
        raise ValueError("sync rows are empty")
    for row_index, row in enumerate(rows):
        frame = int(row["frame_index"])
        sample = int(row["sample_index"])
        view_id = str(row["view_id"])
        step = int(row["global_step"])
        phase = str(row["phase"])
        phase_step = int(row["phase_step"])
        if frame != previous_frame + 1:
            raise ValueError("sync frame indices must be contiguous from zero")
        expected_sample = row_index // len(view_ids)
        expected_view = view_ids[row_index % len(view_ids)]
        if sample != expected_sample or view_id != expected_view:
            raise ValueError(
                "each sample must contain the complete fixed view order"
            )
        if phase not in CAPTURE_PHASES:
            raise ValueError(f"unexpected synchronized capture phase: {phase}")
        if phase_step <= 0:
            raise ValueError("sync phase steps must be positive")
        if view_id == view_ids[0]:
            if step <= previous_sample_step:
                raise ValueError(
                    "sync sample global steps must be strictly increasing"
                )
            samples.append(row)
            seen_phases.add(phase)
            previous_sample_step = step
        else:
            first = samples[-1]
            for field in (
                "global_step",
                "phase",
                "phase_step",
                "simulation_time_s",
            ):
                if str(row[field]) != str(first[field]):
                    raise ValueError(
                        "views in one sample differ in physics mapping"
                    )
        previous_frame = frame
    if len(rows) % len(view_ids) != 0:
        raise ValueError("final synchronized sample is missing a view")
    missing = [phase for phase in CAPTURE_PHASES if phase not in seen_phases]
    if missing:
        raise ValueError("capture is missing phases: " + ",".join(missing))
    if sampling["physics_steps_per_frame"] <= 0:
        raise ValueError("invalid sampling contract")
    if phase_step_totals is not None:
        for phase in CAPTURE_PHASES:
            observed = [
                int(row["phase_step"])
                for row in samples
                if row["phase"] == phase
            ]
            expected = expected_sampled_phase_steps(
                int(phase_step_totals[phase]),
                sampling["physics_steps_per_frame"],
            )
            if observed != expected:
                raise ValueError(
                    f"capture schedule is incomplete for {phase}: "
                    f"{observed!r} != {expected!r}"
                )
    return {
        "first_global_step": int(samples[0]["global_step"]),
        "frame_count": len(rows),
        "frames_per_view": {
            view_id: len(samples) for view_id in view_ids
        },
        "last_global_step": int(samples[-1]["global_step"]),
        "phases": list(CAPTURE_PHASES),
        "sample_count": len(samples),
        "view_order": list(view_ids),
    }


def build_hash_manifest(
    *,
    output_directory,
    sync_rows,
    sampling,
    camera_rig,
    render_settings,
    cleanup,
    physics_evidence,
    capture_source,
    phase_step_totals=None,
):
    """Build a manifest after sync CSV and every RGB frame are closed."""

    output = Path(output_directory)
    sync_path = output / "video_frame_sync.csv"
    validation = validate_sync_rows(
        sync_rows, sampling, phase_step_totals=phase_step_totals
    )
    frame_hashes = {
        str(row["rgb_filename"]): sha256_file(output / row["rgb_filename"])
        for row in sync_rows
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "camera_rig": camera_rig,
        "capture_source": dict(capture_source),
        "cleanup": cleanup,
        "frame_capture": validation,
        "frame_files_sha256": frame_hashes,
        "physics_evidence": physics_evidence,
        "render_settings": render_settings,
        "sampling": sampling,
        "step_mapping_semantics": (
            "all fixed views read in manifest priority immediately after "
            "world.step(render=True), with no intervening physics step or "
            "pose write; all map to the same most recently completed "
            "physics global_step"
        ),
        "sync_columns": list(SYNC_FIELDS),
        "sync_csv": "video_frame_sync.csv",
        "sync_csv_sha256": sha256_file(sync_path),
    }


class D38999ToothSyncCapture:
    """Own four fixed RGB RenderProducts and one hash-bound step sidecar."""

    def __init__(
        self,
        *,
        output_directory,
        physics_rate_hz,
        bindings,
        stage,
        render_settings,
        capture_rate_hz=DEFAULT_CAPTURE_RATE_HZ,
        resolution=DEFAULT_RESOLUTION,
        camera_eye_m=DEFAULT_CAMERA_EYE_M,
        camera_target_m=DEFAULT_CAMERA_TARGET_M,
    ):
        required = {"Gf", "Image", "UsdGeom", "rep"}
        missing = sorted(required - set(bindings))
        if missing:
            raise ValueError(f"missing tooth capture bindings: {missing}")
        self._Gf = bindings["Gf"]
        self._Image = bindings["Image"]
        self._UsdGeom = bindings["UsdGeom"]
        self._rep = bindings["rep"]
        self._stage = stage
        # Snapshot provenance before any long GPU work.  Reading __file__ only
        # during finalize could falsely bind a run to code edited mid-capture.
        source_path = _MODULE_SOURCE_PATH
        self._capture_source = {
            "path": str(source_path),
            "sha256_at_import": _MODULE_SHA256_AT_IMPORT,
            "sha256_at_start": sha256_file(source_path),
        }
        self._output = prepare_empty_output_directory(output_directory)
        self._frames = self._output / "frames"
        self._frames.mkdir(parents=True, exist_ok=True)
        self._sampling = validate_sampling_rates(
            physics_rate_hz, capture_rate_hz
        )
        self._camera_rig = build_camera_rig_contract(
            resolution, camera_eye_m, camera_target_m
        )
        self._resolution = tuple(
            self._camera_rig["views"][0]["resolution"]
        )
        self._render_settings = dict(render_settings)
        self._view_resources = []
        self._rows = []
        self._closed = False
        self._stream = (self._output / "video_frame_sync.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._writer = csv.DictWriter(self._stream, fieldnames=SYNC_FIELDS)
        self._writer.writeheader()

        # Check the complete namespace before authoring any camera, so a stale
        # later prim cannot leave newly-authored earlier prims behind.
        for view in self._camera_rig["views"]:
            camera_prim = stage.GetPrimAtPath(view["prim_path"])
            if camera_prim is not None and camera_prim.IsValid():
                raise RuntimeError(
                    f"tooth sync camera path already exists: "
                    f"{view['prim_path']}"
                )
        for view in self._camera_rig["views"]:
            suffix = {
                "rear_left": "RearLeft",
                "rear_right": "RearRight",
                "front_left": "FrontLeft",
                "front_right": "FrontRight",
            }[view["view_id"]]
            camera_prim = self._rep.functional.create.camera(
                position=tuple(view["eye_m"]),
                look_at=tuple(view["target_m"]),
                focal_length=view["focal_length_mm"],
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
                name=f"ToothSyncCamera{suffix}",
                parent="/World/D38999NutRegrasp",
            )
            if str(camera_prim.GetPath()) != view["prim_path"]:
                raise RuntimeError(
                    "tooth sync camera path differs from contract"
                )
            camera = self._UsdGeom.Camera(camera_prim)
            camera.CreateVerticalApertureAttr().Set(
                20.955
                * float(self._resolution[1])
                / float(self._resolution[0])
            )
            render_product = self._rep.create.render_product(
                camera_prim,
                self._resolution,
                name=view["render_product_name"],
            )
            annotator = self._rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([render_product.path])
            self._view_resources.append(
                {
                    "annotator": annotator,
                    "camera_path": view["prim_path"],
                    "render_product": render_product,
                    "view_id": view["view_id"],
                }
            )
            (self._frames / view["view_id"]).mkdir(parents=True)

    def maybe_capture(self, *, global_step, phase, phase_step):
        """Capture RGB after this completed physics step when scheduled."""

        if self._closed:
            raise RuntimeError("tooth sync capture is already finalized")
        if not should_capture_phase_step(
            phase,
            phase_step,
            physics_steps_per_frame=self._sampling[
                "physics_steps_per_frame"
            ],
        ):
            return False
        # World.step(render=True) already presented every RenderProduct.  Read
        # and validate all views before saving any, with no intervening physics
        # advancement or pose write, so a logical sample cannot be half-valid.
        images = []
        for resource in self._view_resources:
            rgba = resource["annotator"].get_data()
            if rgba is None or getattr(rgba, "ndim", 0) != 3:
                raise RuntimeError(
                    f"RGB annotator returned no frame for "
                    f"{resource['view_id']}"
                )
            if rgba.shape[0:2] != (
                self._resolution[1],
                self._resolution[0],
            ) or rgba.shape[2] not in (3, 4):
                raise RuntimeError(
                    f"RGB annotator shape differs for {resource['view_id']}"
                )
            images.append(
                (resource["view_id"], rgba[:, :, :3].copy(), time.monotonic())
            )
        sample_index = len(self._rows) // len(self._view_resources)
        for view_id, rgb, timestamp in images:
            frame_index = len(self._rows)
            filename = (
                f"frames/{view_id}/frame_{sample_index:06d}.png"
            )
            self._Image.fromarray(rgb).save(self._output / filename)
            row = {
                "frame_index": frame_index,
                "sample_index": sample_index,
                "view_id": view_id,
                "global_step": int(global_step),
                "phase": str(phase),
                "phase_step": int(phase_step),
                "simulation_time_s": float(global_step)
                / float(self._sampling["physics_rate_hz"]),
                "timestamp_s": timestamp,
                "rgb_filename": filename,
            }
            self._writer.writerow(row)
            self._rows.append(row)
        self._stream.flush()
        return True

    def finalize(self, *, physics_report_path, physics_summary_path):
        """Close evidence, release render resources and hash all files."""

        if self._closed:
            manifest_path = self._output / "video_capture_manifest.json"
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        self._closed = True
        self._stream.close()
        cleanup_errors = []
        detached = 0
        destroyed = 0
        removed = 0
        for resource in self._view_resources:
            view_id = resource["view_id"]
            try:
                resource["annotator"].detach()
                detached += 1
            except Exception as exception:  # pragma: no cover - Isaac boundary
                cleanup_errors.append(
                    f"{view_id}.annotator.detach:"
                    f"{type(exception).__name__}:{exception}"
                )
            try:
                resource["render_product"].destroy()
                destroyed += 1
            except Exception as exception:  # pragma: no cover - Isaac boundary
                cleanup_errors.append(
                    f"{view_id}.render_product.destroy:"
                    f"{type(exception).__name__}:{exception}"
                )
            try:
                if self._stage.RemovePrim(resource["camera_path"]):
                    removed += 1
            except Exception as exception:  # pragma: no cover - Isaac boundary
                cleanup_errors.append(
                    f"{view_id}.camera.RemovePrim:"
                    f"{type(exception).__name__}:{exception}"
                )
        view_count = len(self._camera_rig["views"])
        annotator_detached = detached == view_count
        camera_prim_removed = removed == view_count
        render_product_destroyed = destroyed == view_count
        cleanup = {
            "annotator_detached": annotator_detached,
            "annotators_detached_count": detached,
            "camera_prim_removed": camera_prim_removed,
            "camera_prims_removed_count": removed,
            "errors": cleanup_errors,
            "object_pose_writes": 0,
            "render_product_destroyed": render_product_destroyed,
            "render_products_destroyed_count": destroyed,
            "resources_released": bool(
                annotator_detached
                and camera_prim_removed
                and render_product_destroyed
                and not cleanup_errors
            ),
            "stage_cleared": False,
            "view_count": view_count,
            "world_reset": False,
        }
        report_path = Path(physics_report_path)
        report_document = json.loads(report_path.read_text(encoding="utf-8"))
        color_report = report_document.get("color_identification", {})
        if color_report.get("authored_in_session_layer") is not True:
            raise RuntimeError(
                "physics report does not prove 24-tooth color IDs were "
                "authored"
            )
        phase_step_totals = report_document.get("phase_steps", {})
        if any(phase not in phase_step_totals for phase in CAPTURE_PHASES):
            raise RuntimeError(
                "physics report is missing capture phase totals"
            )
        physics_evidence = {
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "summary_path": str(Path(physics_summary_path).resolve()),
            "summary_sha256": sha256_file(physics_summary_path),
            "tooth_color_ids_authored": True,
            "tooth_color_id_count": len(color_report.get("colors_rgb", {})),
        }
        if physics_evidence["tooth_color_id_count"] != 24:
            raise RuntimeError(
                "physics report does not contain 24 tooth colors"
            )
        manifest = build_hash_manifest(
            output_directory=self._output,
            sync_rows=self._rows,
            sampling=self._sampling,
            camera_rig=self._camera_rig,
            render_settings=self._render_settings,
            cleanup=cleanup,
            physics_evidence=physics_evidence,
            capture_source={
                **self._capture_source,
                "sha256_at_finalize": sha256_file(
                    self._capture_source["path"]
                ),
                "unchanged_during_capture": bool(
                    self._capture_source["sha256_at_import"]
                    == self._capture_source["sha256_at_start"]
                    == sha256_file(self._capture_source["path"])
                ),
            },
            phase_step_totals=phase_step_totals,
        )
        manifest["passed"] = bool(
            cleanup["resources_released"]
            and manifest["capture_source"]["unchanged_during_capture"]
        )
        manifest_path = self._output / "video_capture_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
