#!/usr/bin/env python3

"""Independent two-view axial supplement for the prepared tooth capture.

This module deliberately leaves ``d38999_tooth_sync_capture.py`` and the
prepared regrasp runner byte-for-byte unchanged.  A small opt-in launcher
installs the combined class exported here only for that process.  The class
delegates the original four-camera work to the original capture class and
adds two presentation-only, steep-oblique cameras.  No method advances the
timeline or writes an object, robot, constraint, or physics pose.

The two added views target the only identities still absent from the
four-view ghost-fingers union: Segment_13 at 195 degrees and Segment_23 at
345 degrees.  Their horizontal sight azimuths are 225 and 315 degrees,
respectively, while a 50.8 degree elevation exposes the tooth top faces much
more strongly than the original shallow-oblique cameras.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import d38999_tooth_sync_capture as _base_capture


SCHEMA_VERSION = "kcg_d38999_tooth_axial_capture_v1"
DEFAULT_RESOLUTION = (960, 720)
DEFAULT_CAMERA_TARGET_M = (0.55, 0.185, 0.280)
VIEW_IDS = ("axial_segment13", "axial_segment23")
VIEW_DEFINITIONS = (
    {
        "eye_m": (0.44, 0.075, 0.47),
        "segment_angle_degrees": 195.0,
        "target_segment": "Segment_13",
        "view_id": "axial_segment13",
    },
    {
        "eye_m": (0.66, 0.075, 0.47),
        "segment_angle_degrees": 345.0,
        "target_segment": "Segment_23",
        "view_id": "axial_segment23",
    },
)
CAPTURE_PHASES = _base_capture.CAPTURE_PHASES
SYNC_FIELDS = _base_capture.SYNC_FIELDS
AXIAL_MANIFEST_NAME = "axial_capture_manifest.json"
AXIAL_GHOST_BUNDLE_NAME = "axial_ghost_bundle_manifest.json"
AXIAL_GHOST_BUNDLE_SCHEMA_VERSION = (
    "kcg_d38999_tooth_axial_ghost_bundle_v1"
)


def sha256_file(path):
    """Return the SHA-256 of one provenance or evidence file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path):
    """Bind an existing file by absolute path, byte size and SHA-256."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


_MODULE_SOURCE_PATH = Path(__file__).resolve()
_MODULE_SHA256_AT_IMPORT = sha256_file(_MODULE_SOURCE_PATH)
_EXTENSION_CONFIG = None


def _unit_sight_vector(eye, target):
    vector = tuple(float(a) - float(b) for a, b in zip(eye, target))
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("axial camera eye must differ from its target")
    return tuple(value / norm for value in vector)


def _target_exposure(eye, target, segment_angle_degrees):
    """Return analytic face exposure for one target tooth and camera.

    The dot products are a planning contract, not image evidence.  A positive
    radial value means the camera sees the segment's outer face; the axial
    value measures top-face exposure.  They are stored in the run manifest so
    a later camera edit cannot silently invalidate the geometric rationale.
    """

    sight = _unit_sight_vector(eye, target)
    angle = math.radians(float(segment_angle_degrees))
    radial = (math.cos(angle), math.sin(angle), 0.0)
    radial_cosine = sum(a * b for a, b in zip(sight, radial))
    axial_cosine = sight[2]
    horizontal = math.hypot(sight[0], sight[1])
    elevation = math.degrees(math.atan2(sight[2], horizontal))
    azimuth = math.degrees(math.atan2(sight[1], sight[0])) % 360.0
    return {
        "axial_top_face_cosine": axial_cosine,
        "camera_azimuth_degrees": azimuth,
        "camera_elevation_degrees": elevation,
        "radial_outer_face_cosine": radial_cosine,
        "range_m": math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(eye, target))
        ),
    }


def build_axial_camera_rig_contract(
    resolution=DEFAULT_RESOLUTION,
    target=DEFAULT_CAMERA_TARGET_M,
):
    """Build the exact fixed two-camera geometry used by the supplement."""

    resolution = tuple(int(value) for value in resolution)
    target = tuple(float(value) for value in target)
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("axial resolution must be positive width/height")
    if len(target) != 3 or not all(math.isfinite(value) for value in target):
        raise ValueError("axial camera target must be a finite 3-vector")
    suffixes = {
        "axial_segment13": "AxialSegment13",
        "axial_segment23": "AxialSegment23",
    }
    views = []
    for definition in VIEW_DEFINITIONS:
        eye = tuple(float(value) for value in definition["eye_m"])
        exposure = _target_exposure(
            eye, target, definition["segment_angle_degrees"]
        )
        if (
            exposure["axial_top_face_cosine"] < 0.70
            or exposure["radial_outer_face_cosine"] < 0.50
        ):
            raise ValueError("axial target-face exposure is too weak")
        view_id = definition["view_id"]
        suffix = suffixes[view_id]
        views.append(
            {
                "analytic_target_exposure": exposure,
                "eye_m": list(eye),
                "fixed_before_play": True,
                "focal_length_mm": 50.0,
                "prim_path": (
                    "/World/D38999NutRegrasp/"
                    f"ToothSyncCamera{suffix}"
                ),
                "render_product_name": (
                    f"D38999ToothSyncRenderProduct{suffix}"
                ),
                "resolution": list(resolution),
                "segment_angle_degrees": float(
                    definition["segment_angle_degrees"]
                ),
                "target_m": list(target),
                "target_segment": definition["target_segment"],
                "view_id": view_id,
            }
        )
    return {
        "axis_semantics": "prepared CouplingNut local +Z equals world +Z",
        "same_completed_physics_step_as_base_four_views": True,
        "view_priority": list(VIEW_IDS),
        "views": views,
    }


def validate_sync_rows(rows, sampling, phase_step_totals=None):
    """Fail closed unless every sample has both views and exact scheduling."""

    if not rows or len(rows) % len(VIEW_IDS):
        raise ValueError("axial sync rows are empty or incomplete")
    samples = []
    previous_global_step = -1
    previous_frame = -1
    seen_phases = set()
    for row_index, row in enumerate(rows):
        frame = int(row["frame_index"])
        sample_index = int(row["sample_index"])
        view_id = str(row["view_id"])
        if frame != previous_frame + 1:
            raise ValueError("axial frame indices must be contiguous")
        expected_sample = row_index // len(VIEW_IDS)
        expected_view = VIEW_IDS[row_index % len(VIEW_IDS)]
        if sample_index != expected_sample or view_id != expected_view:
            raise ValueError("axial sample view order differs from contract")
        if view_id == VIEW_IDS[0]:
            global_step = int(row["global_step"])
            if global_step <= previous_global_step:
                raise ValueError("axial sample steps must strictly increase")
            if row["phase"] not in CAPTURE_PHASES:
                raise ValueError("unexpected axial capture phase")
            if int(row["phase_step"]) <= 0:
                raise ValueError("axial phase step must be positive")
            samples.append(row)
            seen_phases.add(row["phase"])
            previous_global_step = global_step
        else:
            first = samples[-1]
            for field in (
                "global_step",
                "phase",
                "phase_step",
                "simulation_time_s",
            ):
                if str(row[field]) != str(first[field]):
                    raise ValueError("axial views differ in physics mapping")
        previous_frame = frame
    missing = [phase for phase in CAPTURE_PHASES if phase not in seen_phases]
    if missing:
        raise ValueError(
            "axial capture is missing phases: " + ",".join(missing)
        )
    if phase_step_totals is not None:
        interval = int(sampling["physics_steps_per_frame"])
        for phase in CAPTURE_PHASES:
            observed = [
                int(row["phase_step"])
                for row in samples
                if row["phase"] == phase
            ]
            expected = _base_capture.expected_sampled_phase_steps(
                int(phase_step_totals[phase]), interval
            )
            if observed != expected:
                raise ValueError(
                    f"axial capture schedule is incomplete for {phase}"
                )
    return {
        "first_global_step": int(samples[0]["global_step"]),
        "frame_count": len(rows),
        "frames_per_view": {
            view_id: len(samples) for view_id in VIEW_IDS
        },
        "last_global_step": int(samples[-1]["global_step"]),
        "phases": list(CAPTURE_PHASES),
        "sample_count": len(samples),
        "view_order": list(VIEW_IDS),
    }


def _sample_keys(rows, view_ids):
    first = view_ids[0]
    return [
        (
            int(row["global_step"]),
            str(row["phase"]),
            int(row["phase_step"]),
        )
        for row in rows
        if row["view_id"] == first
    ]


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SYNC_FIELDS:
            raise RuntimeError("capture sync columns differ from contract")
        return list(reader)


def configure_axial_extension(
    *, output_directory, wrapper_source_path, runner_source_path
):
    """Configure the opt-in process before importing the original runner."""

    global _EXTENSION_CONFIG
    if _EXTENSION_CONFIG is not None:
        raise RuntimeError("axial extension was already configured")
    output = Path(output_directory).expanduser().resolve()
    wrapper = Path(wrapper_source_path).expanduser().resolve()
    runner = Path(runner_source_path).expanduser().resolve()
    if not wrapper.is_file() or not runner.is_file():
        raise FileNotFoundError("axial wrapper or prepared runner is missing")
    _EXTENSION_CONFIG = {
        "output_directory": output,
        "runner_source_path": runner,
        "wrapper_source_path": wrapper,
    }
    return dict(_EXTENSION_CONFIG)


def _validate_file_binding(binding, expected_path, label):
    """Reject a path, size or hash substitution in a completed sidecar."""

    target = Path(expected_path).expanduser().resolve()
    if Path(binding.get("path", "")).resolve() != target:
        raise RuntimeError(f"{label} path differs")
    expected = file_binding(target)
    if any(binding.get(key) != expected[key] for key in expected):
        raise RuntimeError(f"{label} size/SHA differs")
    return expected


def finalize_axial_ghost_bundle(*, axial_output, ghost_output):
    """Bind the post-run ghost proof to the already-closed axial evidence."""

    axial_root = Path(axial_output).expanduser().resolve()
    ghost_root = Path(ghost_output).expanduser().resolve()
    axial_path = axial_root / AXIAL_MANIFEST_NAME
    ghost_path = ghost_root / "manifest.json"
    sidecar_path = ghost_root / "visibility_sidecar.json"
    axial = json.loads(axial_path.read_text(encoding="utf-8"))
    ghost = json.loads(ghost_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if axial.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("axial manifest schema differs")
    if axial.get("passed") is not True:
        raise RuntimeError("axial manifest is not passed")
    if ghost.get("schema_version") != "kcg_d38999_tooth_ghost_manifest_v1":
        raise RuntimeError("ghost manifest schema differs")
    _validate_file_binding(
        ghost.get("outputs", {}).get("visibility_sidecar", {}),
        sidecar_path,
        "ghost visibility sidecar",
    )
    if (
        sidecar.get("passed") is not True
        or sidecar.get("active") is not False
        or sidecar.get("cleanup", {}).get(
            "session_visibility_opinions_removed"
        )
        is not True
        or sidecar.get("cleanup", {}).get(
            "effective_visibility_restored_to_pre_author_state"
        )
        is not True
    ):
        raise RuntimeError("ghost visibility lifecycle is not passed")
    mutation = sidecar.get("mutation_audit", {})
    if any(
        mutation.get(key) != 0
        for key in (
            "collision_api_writes",
            "material_writes",
            "object_pose_writes",
            "physics_api_writes",
            "xform_writes",
        )
    ):
        raise RuntimeError("ghost sidecar contains a forbidden write")
    base_binding = axial["base_four_view_binding"]
    ghost_base = ghost.get("inputs", {}).get("capture_manifest", {})
    if (
        ghost_base.get("sha256") != base_binding.get("sha256")
        or ghost_base.get("size_bytes") != base_binding.get("size_bytes")
        or Path(ghost_base.get("path", "")).resolve()
        != Path(base_binding.get("path", "")).resolve()
    ):
        raise RuntimeError("axial and ghost base capture bindings differ")
    runner = axial["provenance"]["prepared_runner"]
    ghost_runner = ghost.get("sources", {}).get("prepared_tooth_runner", {})
    if (
        runner.get("sha256") != ghost_runner.get("sha256")
        or runner.get("size_bytes") != ghost_runner.get("size_bytes")
        or Path(runner.get("path", "")).resolve()
        != Path(ghost_runner.get("path", "")).resolve()
    ):
        raise RuntimeError("axial and ghost runner bindings differ")
    bundle = {
        "schema_version": AXIAL_GHOST_BUNDLE_SCHEMA_VERSION,
        "inputs": {
            "axial_capture_manifest": file_binding(axial_path),
            "base_four_view_capture_manifest": dict(base_binding),
            "ghost_manifest": file_binding(ghost_path),
            "ghost_visibility_sidecar": file_binding(sidecar_path),
        },
        "same_base_capture": True,
        "same_prepared_runner": True,
        "visibility_only_zero_physics_or_pose_writes": True,
        "passed": True,
    }
    path = axial_root / AXIAL_GHOST_BUNDLE_NAME
    path.write_text(
        json.dumps(bundle, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


class D38999ToothAxialCapture:
    """Own two fixed RGB RenderProducts and their independent sidecar."""

    def __init__(
        self,
        *,
        output_directory,
        physics_rate_hz,
        bindings,
        stage,
        render_settings,
        runner_source_path,
        wrapper_source_path,
        capture_rate_hz=_base_capture.DEFAULT_CAPTURE_RATE_HZ,
        resolution=DEFAULT_RESOLUTION,
        camera_target_m=DEFAULT_CAMERA_TARGET_M,
    ):
        required = {"Image", "UsdGeom", "rep"}
        missing = sorted(required - set(bindings))
        if missing:
            raise ValueError(f"missing axial capture bindings: {missing}")
        self._Image = bindings["Image"]
        self._UsdGeom = bindings["UsdGeom"]
        self._rep = bindings["rep"]
        self._stage = stage
        self._source_at_start = sha256_file(_MODULE_SOURCE_PATH)
        self._runner_source = Path(runner_source_path).resolve()
        self._wrapper_source = Path(wrapper_source_path).resolve()
        self._runner_at_start = sha256_file(self._runner_source)
        self._wrapper_at_start = sha256_file(self._wrapper_source)
        self._output = _base_capture.prepare_empty_output_directory(
            output_directory
        )
        self._frames = self._output / "frames"
        self._frames.mkdir(parents=True, exist_ok=True)
        self._sampling = _base_capture.validate_sampling_rates(
            physics_rate_hz, capture_rate_hz
        )
        self._camera_rig = build_axial_camera_rig_contract(
            resolution, camera_target_m
        )
        self._resolution = tuple(resolution)
        self._render_settings = dict(render_settings)
        self._resources = []
        self._rows = []
        self._closed = False
        self._stream = (self._output / "video_frame_sync.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._writer = csv.DictWriter(self._stream, fieldnames=SYNC_FIELDS)
        self._writer.writeheader()
        try:
            self._create_resources()
        except Exception:
            self.cleanup()
            raise

    def _create_resources(self):
        for view in self._camera_rig["views"]:
            prim = self._stage.GetPrimAtPath(view["prim_path"])
            if prim is not None and prim.IsValid():
                raise RuntimeError(
                    f"axial camera path already exists: {view['prim_path']}"
                )
        for view in self._camera_rig["views"]:
            suffix = {
                "axial_segment13": "AxialSegment13",
                "axial_segment23": "AxialSegment23",
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
                raise RuntimeError("axial camera path differs from contract")
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
            self._resources.append(
                {
                    "annotator": annotator,
                    "camera_path": view["prim_path"],
                    "render_product": render_product,
                    "view_id": view["view_id"],
                }
            )
            (self._frames / view["view_id"]).mkdir(parents=True)

    def maybe_capture(self, *, global_step, phase, phase_step):
        """Read both rendered views after the caller's completed step."""

        if self._closed:
            raise RuntimeError("axial capture is already finalized")
        if not _base_capture.should_capture_phase_step(
            phase,
            phase_step,
            physics_steps_per_frame=self._sampling[
                "physics_steps_per_frame"
            ],
        ):
            return False
        images = []
        for resource in self._resources:
            rgba = resource["annotator"].get_data()
            if rgba is None or getattr(rgba, "ndim", 0) != 3:
                raise RuntimeError(
                    f"axial RGB missing for {resource['view_id']}"
                )
            if rgba.shape[0:2] != (
                self._resolution[1],
                self._resolution[0],
            ) or rgba.shape[2] not in (3, 4):
                raise RuntimeError("axial RGB shape differs from contract")
            images.append(
                (resource["view_id"], rgba[:, :, :3].copy(), time.monotonic())
            )
        sample_index = len(self._rows) // len(VIEW_IDS)
        for view_id, rgb, timestamp in images:
            filename = f"frames/{view_id}/frame_{sample_index:06d}.png"
            self._Image.fromarray(rgb).save(self._output / filename)
            row = {
                "frame_index": len(self._rows),
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

    def cleanup(self):
        """Release only axial presentation resources; safe after failures."""

        if not self._stream.closed:
            self._stream.close()
        errors = []
        detached = destroyed = removed = 0
        for resource in self._resources:
            try:
                resource["annotator"].detach()
                detached += 1
            except Exception as exception:  # pragma: no cover - Isaac edge
                errors.append(
                    f"{resource['view_id']}.detach:"
                    f"{type(exception).__name__}:{exception}"
                )
            try:
                resource["render_product"].destroy()
                destroyed += 1
            except Exception as exception:  # pragma: no cover - Isaac edge
                errors.append(
                    f"{resource['view_id']}.destroy:"
                    f"{type(exception).__name__}:{exception}"
                )
            try:
                if self._stage.RemovePrim(resource["camera_path"]):
                    removed += 1
            except Exception as exception:  # pragma: no cover - Isaac edge
                errors.append(
                    f"{resource['view_id']}.RemovePrim:"
                    f"{type(exception).__name__}:{exception}"
                )
        # Emptying the list makes this method genuinely idempotent.
        resource_count = len(self._resources)
        self._resources = []
        return {
            "annotators_detached_count": detached,
            "camera_prims_removed_count": removed,
            "errors": errors,
            "object_pose_writes": 0,
            "physics_steps": 0,
            "render_products_destroyed_count": destroyed,
            "resources_released": bool(
                resource_count == len(VIEW_IDS)
                and detached == resource_count
                and destroyed == resource_count
                and removed == resource_count
                and not errors
            ),
            "view_count": resource_count,
        }

    def finalize(
        self,
        *,
        physics_report_path,
        physics_summary_path,
        base_capture_manifest_path,
    ):
        """Close, validate and hash-bind the axial supplement."""

        if self._closed:
            path = self._output / AXIAL_MANIFEST_NAME
            return json.loads(path.read_text(encoding="utf-8"))
        self._closed = True
        cleanup = self.cleanup()
        report_path = Path(physics_report_path).resolve()
        summary_path = Path(physics_summary_path).resolve()
        base_manifest_path = Path(base_capture_manifest_path).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        phase_totals = report.get("phase_steps", {})
        validation = validate_sync_rows(
            self._rows, self._sampling, phase_step_totals=phase_totals
        )
        base_manifest = json.loads(
            base_manifest_path.read_text(encoding="utf-8")
        )
        if (
            base_manifest.get("schema_version")
            != _base_capture.SCHEMA_VERSION
            or base_manifest.get("passed") is not True
        ):
            raise RuntimeError("base four-view manifest is not passed")
        base_rows = _read_csv(
            base_manifest_path.parent / "video_frame_sync.csv"
        )
        base_view_ids = tuple(
            base_manifest["camera_rig"]["view_priority"]
        )
        sync_exact = _sample_keys(base_rows, base_view_ids) == _sample_keys(
            self._rows, VIEW_IDS
        )
        if not sync_exact:
            raise RuntimeError("axial and base sample keys differ")
        frame_hashes = {
            row["rgb_filename"]: sha256_file(
                self._output / row["rgb_filename"]
            )
            for row in self._rows
        }
        source_at_finalize = sha256_file(_MODULE_SOURCE_PATH)
        runner_at_finalize = sha256_file(self._runner_source)
        wrapper_at_finalize = sha256_file(self._wrapper_source)
        provenance_unchanged = bool(
            _MODULE_SHA256_AT_IMPORT
            == self._source_at_start
            == source_at_finalize
            and self._runner_at_start == runner_at_finalize
            and self._wrapper_at_start == wrapper_at_finalize
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "base_four_view_binding": file_binding(base_manifest_path),
            "camera_rig": self._camera_rig,
            "capture_source": {
                "path": str(_MODULE_SOURCE_PATH),
                "sha256_at_finalize": source_at_finalize,
                "sha256_at_import": _MODULE_SHA256_AT_IMPORT,
                "sha256_at_start": self._source_at_start,
            },
            "cleanup": cleanup,
            "frame_capture": validation,
            "frame_files_sha256": frame_hashes,
            "physics_evidence": {
                "report": file_binding(report_path),
                "summary": file_binding(summary_path),
            },
            "provenance": {
                "prepared_runner": file_binding(self._runner_source),
                "runner_sha256_at_start": self._runner_at_start,
                "wrapper": file_binding(self._wrapper_source),
                "wrapper_sha256_at_start": self._wrapper_at_start,
                "unchanged_during_capture": provenance_unchanged,
            },
            "render_settings": self._render_settings,
            "sampling": self._sampling,
            "same_sample_keys_as_base_four_views": sync_exact,
            "step_mapping_semantics": (
                "both axial annotators are read after the same completed "
                "physics step and after the base four annotators, with no "
                "intervening physics step or pose write"
            ),
            "sync_columns": list(SYNC_FIELDS),
            "sync_csv": "video_frame_sync.csv",
            "sync_csv_sha256": sha256_file(
                self._output / "video_frame_sync.csv"
            ),
        }
        manifest["passed"] = bool(
            cleanup["resources_released"]
            and provenance_unchanged
            and sync_exact
        )
        manifest_path = self._output / AXIAL_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if manifest["passed"] is not True:
            raise RuntimeError("axial capture evidence failed")
        return manifest


class D38999ToothSyncCapture:
    """Process-local adapter combining unchanged base and axial views."""

    def __init__(self, **kwargs):
        if _EXTENSION_CONFIG is None:
            raise RuntimeError("axial capture extension is not configured")
        self._base_output = Path(kwargs["output_directory"]).resolve()
        self._axial = D38999ToothAxialCapture(
            output_directory=_EXTENSION_CONFIG["output_directory"],
            physics_rate_hz=kwargs["physics_rate_hz"],
            bindings=kwargs["bindings"],
            stage=kwargs["stage"],
            render_settings=kwargs["render_settings"],
            runner_source_path=_EXTENSION_CONFIG["runner_source_path"],
            wrapper_source_path=_EXTENSION_CONFIG["wrapper_source_path"],
            capture_rate_hz=kwargs.get(
                "capture_rate_hz", _base_capture.DEFAULT_CAPTURE_RATE_HZ
            ),
        )
        try:
            self._base = _base_capture.D38999ToothSyncCapture(**kwargs)
        except Exception:
            self._axial.cleanup()
            raise

    def maybe_capture(self, *, global_step, phase, phase_step):
        base_captured = self._base.maybe_capture(
            global_step=global_step, phase=phase, phase_step=phase_step
        )
        axial_captured = self._axial.maybe_capture(
            global_step=global_step, phase=phase, phase_step=phase_step
        )
        if base_captured != axial_captured:
            raise RuntimeError("base and axial capture schedules diverged")
        return base_captured

    def finalize(self, *, physics_report_path, physics_summary_path):
        base_manifest = self._base.finalize(
            physics_report_path=physics_report_path,
            physics_summary_path=physics_summary_path,
        )
        base_path = self._base_output / "video_capture_manifest.json"
        self._axial.finalize(
            physics_report_path=physics_report_path,
            physics_summary_path=physics_summary_path,
            base_capture_manifest_path=base_path,
        )
        # The original runner and ghost runtime must receive the unchanged
        # four-view manifest, not a look-alike six-view schema.
        return base_manifest


__all__ = [
    "AXIAL_GHOST_BUNDLE_NAME",
    "AXIAL_GHOST_BUNDLE_SCHEMA_VERSION",
    "AXIAL_MANIFEST_NAME",
    "CAPTURE_PHASES",
    "DEFAULT_CAMERA_TARGET_M",
    "DEFAULT_RESOLUTION",
    "D38999ToothAxialCapture",
    "D38999ToothSyncCapture",
    "SCHEMA_VERSION",
    "SYNC_FIELDS",
    "VIEW_DEFINITIONS",
    "VIEW_IDS",
    "build_axial_camera_rig_contract",
    "configure_axial_extension",
    "file_binding",
    "finalize_axial_ghost_bundle",
    "sha256_file",
    "validate_sync_rows",
]
