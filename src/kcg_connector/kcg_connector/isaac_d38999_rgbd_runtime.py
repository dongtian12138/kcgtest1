"""Reusable in-World D38999 RGB-D capture for Isaac Sim 6.0.1.

The module is intentionally importable without Isaac Sim.  Runtime objects and
Isaac/Replicator/USD bindings are injected by the caller, so pure tests and ROS
package imports do not start Kit or acquire GPU resources.  Capture never
clears or resets a World and never writes an endpoint pose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kcg_connector.rgbd_pose_bootstrap import (
    intersect_camera_ray_with_horizontal_plane,
    robust_semantic_mask_center_uv,
    robust_world_xy_centroid,
    semantic_ids_for_label,
    summarize_mask_depth,
)


MINIMUM_ENDPOINT_CENTER_MARGIN_PX = 16
RGBD_CAMERA_FOCAL_LENGTH_MM = 24.0
RGBD_CAMERA_HORIZONTAL_APERTURE_MM = 20.955
RGBD_CAMERA_CLIPPING_RANGE_M = (0.1, 10.0)
RGBD_STAGE_VALUE_ATOL = 1.0e-6


@dataclass(frozen=True)
class D38999RgbdRuntimeCapture:
    """JSON-ready evidence plus endpoint poses used only for validation."""

    metrics: dict[str, Any]
    loose_position_world_m: tuple[float, float, float] | None
    loose_orientation_wxyz: tuple[float, float, float, float] | None
    fixed_position_world_m: tuple[float, float, float] | None
    fixed_orientation_wxyz: tuple[float, float, float, float] | None

    @property
    def passed(self) -> bool:
        return self.metrics.get("passed") is True


def endpoint_projection_records(endpoint_uv, resolution):
    """Return fail-closed in-frame records for projected endpoint centers."""
    if len(resolution) != 2:
        raise ValueError("camera resolution must contain width and height")
    width, height = (int(value) for value in resolution)
    margin = MINIMUM_ENDPOINT_CENTER_MARGIN_PX
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("camera resolution is too small for projection gate")
    records = {}
    for endpoint, raw_uv in endpoint_uv.items():
        if len(raw_uv) != 2:
            raise ValueError(f"{endpoint} projection must contain u and v")
        u_value, v_value = (float(value) for value in raw_uv)
        if not all(math.isfinite(value) for value in (u_value, v_value)):
            raise ValueError(f"{endpoint} projection must be finite")
        records[endpoint] = {
            "in_frame": bool(
                margin <= u_value < width - margin
                and margin <= v_value < height - margin
            ),
            "margin_px": margin,
            "uv_px": [u_value, v_value],
        }
    return records


def validate_real_endpoint_semantic_ids(endpoint_ids, observed_ids):
    """Require endpoint IDs beyond renderer sentinels and present in pixels."""
    observed = {int(value) for value in observed_ids}
    validated = {}
    for endpoint, raw_ids in endpoint_ids.items():
        values = tuple(int(value) for value in raw_ids)
        if not values or any(value in (0, 1) for value in values):
            raise RuntimeError(
                f"{endpoint} uses BACKGROUND/UNLABELLED semantic IDs"
            )
        missing = sorted(set(values) - observed)
        if missing:
            raise RuntimeError(
                f"{endpoint} semantic IDs have no rendered pixels: {missing}"
            )
        validated[endpoint] = values
    return validated


def world_pose(bindings, prim):
    """Read one USD world pose through caller-provided pxr bindings."""
    Gf = bindings["Gf"]
    Usd = bindings["Usd"]
    UsdGeom = bindings["UsdGeom"]
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    quaternion = transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return (
        tuple(float(value) for value in transform.GetTranslation()),
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _prim_at_path(stage, path):
    """Return a valid stage prim or ``None`` without relying on truthiness."""
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        return None
    return prim


def _host_numpy(value, *, dtype):
    """Convert Isaac tensor values on GPU to a host NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _require_prim_type(prim, path, expected_type):
    """Fail closed when a persistent RGB-D prim has an alien USD type."""
    actual_type = str(prim.GetTypeName())
    if actual_type != expected_type:
        raise RuntimeError(
            f"RGB-D prim {path} has type {actual_type!r}; "
            f"expected {expected_type!r}"
        )


def _required_numeric_attribute(prim, path, name):
    """Read one authored numeric USD attribute from a reused scene prim."""
    attribute = prim.GetAttribute(name)
    if attribute is None or not attribute.IsValid():
        raise RuntimeError(f"RGB-D prim {path} is missing attribute {name}")
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"RGB-D prim {path} has no value for {name}")
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exception:
        raise RuntimeError(
            f"RGB-D prim {path} attribute {name} is not numeric"
        ) from exception
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"RGB-D prim {path} attribute {name} is not finite"
        )
    return values


def _validate_numeric_attribute(prim, path, name, expected):
    """Require a persistent attribute to equal the capture contract."""
    actual = _required_numeric_attribute(prim, path, name)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected_array.shape or not np.allclose(
        actual,
        expected_array,
        rtol=0.0,
        atol=RGBD_STAGE_VALUE_ATOL,
    ):
        raise RuntimeError(
            f"RGB-D prim {path} attribute {name} differs from contract: "
            f"{actual.tolist()!r} != {expected_array.tolist()!r}"
        )


def _validate_camera_world_contract(bindings, camera_prim, camera_config):
    """Validate the fixed camera eye and optical-axis target direction."""
    Gf = bindings["Gf"]
    Usd = bindings["Usd"]
    UsdGeom = bindings["UsdGeom"]
    matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    actual_eye = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
    expected_eye = np.asarray(camera_config.eye_m, dtype=np.float64)
    if actual_eye.shape != (3,) or not np.allclose(
        actual_eye,
        expected_eye,
        rtol=0.0,
        atol=RGBD_STAGE_VALUE_ATOL,
    ):
        raise RuntimeError(
            "RGB-D camera world position differs from contract: "
            f"{actual_eye.tolist()!r} != {expected_eye.tolist()!r}"
        )

    expected_forward = (
        np.asarray(camera_config.target_m, dtype=np.float64) - expected_eye
    )
    expected_norm = float(np.linalg.norm(expected_forward))
    if expected_norm <= RGBD_STAGE_VALUE_ATOL:
        raise ValueError("RGB-D camera eye and target must differ")
    expected_forward /= expected_norm
    # A USD camera observes along its local negative Z axis.  Checking that
    # axis prevents silently reusing a correctly positioned but mis-aimed prim.
    actual_forward = np.asarray(
        matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    actual_norm = float(np.linalg.norm(actual_forward))
    if actual_norm <= RGBD_STAGE_VALUE_ATOL:
        raise RuntimeError("RGB-D camera optical axis is degenerate")
    actual_forward /= actual_norm
    if not np.allclose(
        actual_forward,
        expected_forward,
        rtol=0.0,
        atol=RGBD_STAGE_VALUE_ATOL,
    ):
        raise RuntimeError(
            "RGB-D camera optical axis differs from contract: "
            f"{actual_forward.tolist()!r} != {expected_forward.tolist()!r}"
        )


def ensure_d38999_rgbd_stage_prims(
    *,
    bindings: Mapping[str, Any],
    stage,
    tabletop,
    rgbd,
    camera_clipping_range_m=None,
):
    """Create once or strictly reuse fixed RGB-D camera and lighting prims.

    These prims belong to the caller's stage, not to an individual capture.
    Existing prims are never overwritten: their type and authored parameters
    must match the configured contract exactly.  Per-capture render products
    and annotators are intentionally outside this helper.
    """
    Gf = bindings["Gf"]
    UsdGeom = bindings["UsdGeom"]
    UsdLux = bindings["UsdLux"]
    rep = bindings["rep"]
    records = {}

    lighting_root = tabletop.world.root_prim_path + "/RgbdLighting"
    lighting_root_prim = _prim_at_path(stage, lighting_root)
    if lighting_root_prim is None:
        lighting_root_prim = UsdGeom.Xform.Define(
            stage, lighting_root
        ).GetPrim()
        records["lighting_root"] = "created"
    else:
        records["lighting_root"] = "reused"
    _require_prim_type(lighting_root_prim, lighting_root, "Xform")

    dome_path = lighting_root + "/Fill"
    dome_prim = _prim_at_path(stage, dome_path)
    if dome_prim is None:
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(tabletop.render.dome_light_intensity)
        dome.CreateColorAttr(Gf.Vec3f(*tabletop.render.dome_light_color_rgb))
        dome_prim = dome.GetPrim()
        records["dome_light"] = "created"
    else:
        records["dome_light"] = "reused"
    _require_prim_type(dome_prim, dome_path, "DomeLight")
    _validate_numeric_attribute(
        dome_prim,
        dome_path,
        # UsdLux inputs are namespaced attributes on the underlying prim.
        # Reading the schema shorthand ("intensity") works neither for a
        # reused prim nor for direct Prim.GetAttribute validation.
        "inputs:intensity",
        tabletop.render.dome_light_intensity,
    )
    _validate_numeric_attribute(
        dome_prim,
        dome_path,
        "inputs:color",
        tabletop.render.dome_light_color_rgb,
    )

    key_path = lighting_root + "/Key"
    key_prim = _prim_at_path(stage, key_path)
    if key_prim is None:
        key = UsdLux.DistantLight.Define(stage, key_path)
        key.CreateIntensityAttr(tabletop.render.key_light_intensity)
        key.CreateColorAttr(Gf.Vec3f(*tabletop.render.key_light_color_rgb))
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(
            Gf.Vec3f(*tabletop.render.key_light_rotation_degrees_xyz)
        )
        key_prim = key.GetPrim()
        records["key_light"] = "created"
    else:
        records["key_light"] = "reused"
    _require_prim_type(key_prim, key_path, "DistantLight")
    _validate_numeric_attribute(
        key_prim,
        key_path,
        "inputs:intensity",
        tabletop.render.key_light_intensity,
    )
    _validate_numeric_attribute(
        key_prim,
        key_path,
        "inputs:color",
        tabletop.render.key_light_color_rgb,
    )
    _validate_numeric_attribute(
        key_prim,
        key_path,
        "xformOp:rotateXYZ",
        tabletop.render.key_light_rotation_degrees_xyz,
    )

    clipping_range_m = (
        RGBD_CAMERA_CLIPPING_RANGE_M
        if camera_clipping_range_m is None
        else tuple(float(value) for value in camera_clipping_range_m)
    )
    if (
        len(clipping_range_m) != 2
        or not 0.0 < clipping_range_m[0] < clipping_range_m[1]
    ):
        raise ValueError("camera clipping range must satisfy 0 < near < far")
    camera_path = rgbd.camera.prim_path
    camera_prim = _prim_at_path(stage, camera_path)
    if camera_prim is None:
        camera_parent_path, camera_name = camera_path.rsplit("/", 1)
        camera_prim = rep.functional.create.camera(
            position=rgbd.camera.eye_m,
            look_at=rgbd.camera.target_m,
            focal_length=RGBD_CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture=RGBD_CAMERA_HORIZONTAL_APERTURE_MM,
            clipping_range=clipping_range_m,
            name=camera_name,
            parent=camera_parent_path,
        )
        if str(camera_prim.GetPath()) != camera_path:
            raise RuntimeError("Replicator camera path differs from contract")
        width, height = rgbd.camera.resolution
        vertical_aperture = (
            RGBD_CAMERA_HORIZONTAL_APERTURE_MM
            * float(height)
            / float(width)
        )
        UsdGeom.Camera(camera_prim).CreateVerticalApertureAttr().Set(
            vertical_aperture
        )
        records["camera"] = "created"
    else:
        records["camera"] = "reused"

    _require_prim_type(camera_prim, camera_path, "Camera")
    width, height = rgbd.camera.resolution
    camera_contract = {
        "clippingRange": clipping_range_m,
        "focalLength": RGBD_CAMERA_FOCAL_LENGTH_MM,
        "horizontalAperture": RGBD_CAMERA_HORIZONTAL_APERTURE_MM,
        "verticalAperture": (
            RGBD_CAMERA_HORIZONTAL_APERTURE_MM
            * float(height)
            / float(width)
        ),
    }
    for attribute_name, expected_value in camera_contract.items():
        _validate_numeric_attribute(
            camera_prim, camera_path, attribute_name, expected_value
        )
    _validate_camera_world_contract(bindings, camera_prim, rgbd.camera)
    return camera_prim, {
        "camera_path": camera_path,
        "lighting_root_path": lighting_root,
        "prim_lifecycle": records,
    }


def cleanup_rgbd_runtime_resources(annotators, camera, render_product):
    """Release render resources without deleting stage prims or a World."""
    errors = []
    detached = 0
    for annotator in reversed(tuple(annotators)):
        try:
            annotator.detach()
            detached += 1
        except Exception as exception:  # pragma: no cover - runtime boundary
            errors.append(
                f"annotator.detach: {type(exception).__name__}: {exception}"
            )
    camera_destroyed = camera is None
    if camera is not None:
        try:
            # Camera.destroy() first clears its reference to the externally
            # owned render-product path.  That prevents its destructor from
            # reading an invalid render-product attribute after Kit cleanup.
            camera.destroy()
            camera_destroyed = True
        except Exception as exception:  # pragma: no cover - runtime boundary
            errors.append(
                f"camera.destroy: {type(exception).__name__}: {exception}"
            )
    render_product_destroyed = render_product is None
    if render_product is not None:
        try:
            render_product.destroy()
            render_product_destroyed = True
        except Exception as exception:  # pragma: no cover - runtime boundary
            errors.append(
                "render_product.destroy: "
                f"{type(exception).__name__}: {exception}"
            )
    return {
        "annotator_detach_count": detached,
        "camera_destroyed": camera_destroyed,
        "errors": errors,
        "render_product_destroyed": render_product_destroyed,
        "resources_released": bool(
            detached == len(tuple(annotators))
            and camera_destroyed
            and render_product_destroyed
            and not errors
        ),
        "scene_cleared": False,
        "stage_prims_removed": 0,
        "world_reset": False,
    }


def restore_timeline_after_rgbd_capture(
    world, simulation_app, *, was_playing
):
    """Restore the caller's exact timeline state after Replicator capture.

    The capture path deliberately pauses physics while rendering.  This guard
    restores a playing caller and also re-pauses an intentionally paused caller
    if a future Replicator release unexpectedly starts the timeline.
    """
    if not isinstance(was_playing, bool):
        raise ValueError("was_playing must be boolean")
    playing_after_cleanup = bool(world.is_playing())
    restore_attempted = playing_after_cleanup is not was_playing
    if restore_attempted and was_playing:
        world.play()
        simulation_app.update()
    elif restore_attempted:
        world.pause()
        simulation_app.update()
    playing_after_restore = bool(world.is_playing())
    return {
        "playing_after_cleanup": playing_after_cleanup,
        "playing_after_restore": playing_after_restore,
        "playing_before_capture": was_playing,
        "restore_attempted": restore_attempted,
        # Preserve either caller state exactly.  A capture must not silently
        # start a caller that intentionally supplied a paused World either.
        "restored": playing_after_restore is was_playing,
    }


def pause_timeline_for_rgbd_capture(world, simulation_app, *, was_playing):
    """Freeze physics while Replicator renders the preflight frames.

    Isaac 6's synchronous Replicator path is noisy when it renders zero-time
    subframes while a complex PhysX timeline is running.  The preflight occurs
    before the first intentional robot command, so pausing here is both
    deterministic and physically neutral: ``simulation_app.update`` renders
    once but cannot advance physics while the World is paused.
    """
    if not isinstance(was_playing, bool):
        raise ValueError("was_playing must be boolean")
    pause_attempted = bool(was_playing)
    if pause_attempted:
        world.pause()
        simulation_app.update()
    playing_during_capture = bool(world.is_playing())
    return {
        "pause_attempted": pause_attempted,
        "paused_for_capture": not playing_during_capture,
        "playing_during_capture": playing_during_capture,
    }


def _save_capture_artifacts(
    *,
    bindings,
    rgbd,
    output_dir,
    rgba,
    depth,
    semantic,
    loose_mask,
    fixed_mask,
):
    Image = bindings["Image"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    Image.fromarray(rgb).save(output_path / rgbd.output.rgb_filename)
    np.save(output_path / rgbd.output.depth_numpy_filename, depth)
    valid_depth = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid_depth.size == 0:
        raise RuntimeError("camera depth image has no finite pixels")
    near = float(np.percentile(valid_depth, 2.0))
    far = float(np.percentile(valid_depth, 98.0))
    scaled_depth = np.clip(
        (depth - near) / max(far - near, 1.0e-9), 0, 1
    )
    scaled_depth[~np.isfinite(scaled_depth)] = 1.0
    depth_preview = np.asarray(
        255.0 * (1.0 - scaled_depth), dtype=np.uint8
    )
    Image.fromarray(depth_preview).save(
        output_path / rgbd.output.depth_preview_filename
    )
    semantic_preview = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    semantic_preview[loose_mask] = (235, 80, 70)
    semantic_preview[fixed_mask] = (70, 180, 245)
    Image.fromarray(semantic_preview).save(
        output_path / rgbd.output.semantic_preview_filename
    )


def capture_d38999_rgbd_runtime(
    *,
    bindings: Mapping[str, Any],
    simulation_app,
    world,
    stage,
    tabletop,
    rgbd,
    loose_prim,
    fixed_prim,
    body,
    output_dir: Path | str | None = None,
    pose5d_config: Mapping[str, Any] | None = None,
    pose5d_capture_id: str | None = None,
    pose5d_axis_priors: Mapping[str, tuple[float, float, float]] | None = None,
    pose5d_authorization_gates: Mapping[str, float] | None = None,
) -> D38999RgbdRuntimeCapture:
    """Capture both endpoints in the caller's existing World and episode.

    The function may advance Replicator render frames for warm-up.  It does
    not reset/clear the World, apply an articulation action, or author an
    endpoint transform.  Endpoint truth poses are read only to score the
    simulation milestone and are returned separately from the mask-derived
    XY estimate.
    """
    required = {
        "Camera",
        "Gf",
        "Usd",
        "UsdGeom",
        "UsdLux",
        "add_labels",
        "get_labels",
        "rep",
    }
    if output_dir is not None:
        required.add("Image")
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"missing Isaac RGB-D runtime bindings: {missing}")

    Camera = bindings["Camera"]
    Usd = bindings["Usd"]
    UsdGeom = bindings["UsdGeom"]
    add_labels = bindings["add_labels"]
    get_labels = bindings["get_labels"]
    rep = bindings["rep"]

    metrics = {
        "camera_observation_present": False,
        "capture_episode": "caller_world_same_episode",
        "detector_kind": "isaac_renderer_semantic_annotation_bootstrap",
        "foundation_pose_present": False,
        "full_keyed_6d_vision_pose_claimed": False,
        "learned_detector_present": False,
        "masked_rgbd_xy_used_for_control": False,
        "object_pose_writes_after_start": 0,
        "passed": False,
        "real_camera_present": False,
        "world_reset_or_clear_calls": 0,
    }
    camera = None
    render_product = None
    annotators = []
    loose_position = None
    loose_orientation = None
    fixed_position = None
    fixed_orientation = None
    rgba = None
    cleanup = None
    timeline_playing_before = bool(world.is_playing())
    timeline_pause = None
    try:
        timeline_pause = pause_timeline_for_rgbd_capture(
            world,
            simulation_app,
            was_playing=timeline_playing_before,
        )
        metrics["timeline_pause"] = timeline_pause
        if timeline_pause["paused_for_capture"] is not True:
            raise RuntimeError("RGB-D capture could not pause the timeline")

        def label_endpoint(root_prim, label):
            add_labels(
                root_prim,
                labels=[label],
                taxonomy=rgbd.labels.taxonomy,
            )
            count = sum(
                1
                for prim in Usd.PrimRange(root_prim)
                if prim.IsA(UsdGeom.Gprim)
            )
            if count == 0:
                raise RuntimeError(
                    f"endpoint has no visible geometry for label {label!r}"
                )
            authored_labels = get_labels(root_prim).get(
                rgbd.labels.taxonomy, []
            )
            if authored_labels != [label]:
                raise RuntimeError(
                    "USD root semantic label differs from the requested "
                    f"endpoint label: {authored_labels!r}"
                )
            return count

        metrics["loose_inheriting_gprim_count"] = label_endpoint(
            loose_prim, rgbd.labels.loose_plug
        )
        metrics["fixed_inheriting_gprim_count"] = label_endpoint(
            fixed_prim, rgbd.labels.fixed_receptacle
        )
        simulation_app.update()

        camera_prim, stage_prim_lifecycle = ensure_d38999_rgbd_stage_prims(
            bindings=bindings,
            stage=stage,
            tabletop=tabletop,
            rgbd=rgbd,
        )
        metrics["stage_prim_lifecycle"] = stage_prim_lifecycle
        render_product = rep.create.render_product(
            camera_prim,
            rgbd.camera.resolution,
            name="D38999RgbdRenderProduct",
        )
        render_product_path = render_product.path
        camera = Camera(
            prim_path=rgbd.camera.prim_path,
            name="d38999_rgbd_camera",
            frequency=rgbd.camera.frequency_hz,
            resolution=rgbd.camera.resolution,
            render_product_path=render_product_path,
        )
        # Camera is used only for calibrated projection math.  The three
        # Replicator annotators below own frame acquisition, so initializing
        # Camera's separate NEW_FRAME callback and ReferenceTime annotator is
        # unnecessary and would leave an avoidable event lifecycle in the
        # caller's continuing physical episode.

        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={
                "semanticFilter": f"{rgbd.labels.taxonomy}:*",
                "colorize": False,
            },
        )
        annotators.extend(
            (rgb_annotator, depth_annotator, semantic_annotator)
        )
        for annotator in annotators:
            annotator.attach([render_product_path])
        metrics["render_pipeline"] = {
            "annotator_api": "rep.AnnotatorRegistry.get_annotator",
            "annotators": [
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
            ],
            "camera_semantic_wrapper_used": False,
            "camera_prim_author": (
                "rep.functional.create.camera"
                if stage_prim_lifecycle["prim_lifecycle"]["camera"]
                == "created"
                else "reused_fixed_stage_prim"
            ),
            "camera_prim_lifetime": "caller_stage_persistent",
            "render_product_author": "rep.create.render_product",
            "render_product_path": render_product_path,
            "semantic_filter": f"{rgbd.labels.taxonomy}:*",
            "shared_render_product_for_rgb_depth_semantics": True,
            "vertical_aperture_prevalidated_for_square_pixels": True,
        }

        for _ in range(rgbd.camera.warmup_frames):
            # The World was explicitly paused above.  Zero delta and an
            # explicit pause request make each warm-up render-only; neither
            # simulation time nor endpoint state can advance.
            rep.orchestrator.step(
                rt_subframes=4,
                delta_time=0.0,
                pause_timeline=True,
            )

        projection_body_position, _ = body.get_world_pose()
        projection_fixed_position, _ = world_pose(bindings, fixed_prim)
        projection_points = np.asarray(
            [projection_body_position, projection_fixed_position],
            dtype=np.float64,
        )
        projected_uv = np.asarray(
            camera.get_image_coords_from_world_points(projection_points),
            dtype=np.float64,
        )
        projection_records = endpoint_projection_records(
            {
                "loose_plug": projected_uv[0],
                "fixed_receptacle": projected_uv[1],
            },
            rgbd.camera.resolution,
        )
        projection_gate = all(
            record["in_frame"] for record in projection_records.values()
        )
        camera_position, camera_orientation = camera.get_world_pose()
        metrics["camera_projection"] = {
            "camera_orientation_wxyz": np.asarray(
                camera_orientation, dtype=np.float64
            ).tolist(),
            "camera_position_m": np.asarray(
                camera_position, dtype=np.float64
            ).tolist(),
            "endpoint_centers": projection_records,
            "passed": projection_gate,
            "world_points_m": projection_points.tolist(),
        }

        rgba = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
        semantic_document = semantic_annotator.get_data()
        if not isinstance(semantic_document, dict):
            raise RuntimeError("semantic segmentation annotator is missing")
        semantic = np.asarray(semantic_document.get("data"))
        semantic_info = semantic_document.get("info")
        if not isinstance(semantic_info, dict):
            raise RuntimeError("semantic segmentation metadata is missing")
        id_to_labels = semantic_info.get("idToLabels")
        if not isinstance(id_to_labels, dict):
            raise RuntimeError("semantic ID-to-label map is missing")
        metrics["semantic_id_to_labels"] = id_to_labels
        if rgba.shape[:2] != depth.shape or semantic.shape != depth.shape:
            raise RuntimeError("RGB, depth and semantic shapes differ")
        if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
            raise RuntimeError("RGB output must have three or four channels")
        valid_depth_mask = np.isfinite(depth) & (depth > 0.0)
        metrics["camera_frame_diagnostics"] = {
            "depth_valid_fraction": float(np.mean(valid_depth_mask)),
            "rgb_maximum": int(np.max(rgba[:, :, :3])),
            "rgb_minimum": int(np.min(rgba[:, :, :3])),
            "rgb_mean": float(np.mean(rgba[:, :, :3])),
        }

        loose_ids = semantic_ids_for_label(
            id_to_labels, rgbd.labels.taxonomy, rgbd.labels.loose_plug
        )
        fixed_ids = semantic_ids_for_label(
            id_to_labels,
            rgbd.labels.taxonomy,
            rgbd.labels.fixed_receptacle,
        )
        endpoint_ids = validate_real_endpoint_semantic_ids(
            {
                "fixed_receptacle": fixed_ids,
                "loose_plug": loose_ids,
            },
            np.unique(semantic),
        )
        metrics["endpoint_semantic_ids"] = {
            endpoint: list(values)
            for endpoint, values in endpoint_ids.items()
        }
        metrics["observed_semantic_ids"] = [
            int(value) for value in np.unique(semantic)
        ]
        loose_statistics, loose_mask = summarize_mask_depth(
            semantic, depth, loose_ids
        )
        fixed_statistics, fixed_mask = summarize_mask_depth(
            semantic, depth, fixed_ids
        )
        loose_mask_center_uv = robust_semantic_mask_center_uv(loose_mask)
        fixed_mask_center_uv = robust_semantic_mask_center_uv(fixed_mask)
        mask_center_records = endpoint_projection_records(
            {
                "loose_plug": loose_mask_center_uv,
                "fixed_receptacle": fixed_mask_center_uv,
            },
            rgbd.camera.resolution,
        )
        mask_center_pixels = np.asarray(
            [loose_mask_center_uv, fixed_mask_center_uv], dtype=np.float32
        )
        points_on_rays = np.asarray(
            camera.get_world_points_from_image_coords(
                mask_center_pixels, np.ones(2, dtype=np.float32)
            ),
            dtype=np.float64,
        )
        if points_on_rays.shape != (2, 3):
            raise RuntimeError(
                "camera ray points must form a finite 2x3 array"
            )
        estimator = rgbd.position_estimator
        loose_estimated_world = intersect_camera_ray_with_horizontal_plane(
            camera_position,
            points_on_rays[0],
            estimator.loose_plug_registered_model_height_m,
        )
        fixed_estimated_world = intersect_camera_ray_with_horizontal_plane(
            camera_position,
            points_on_rays[1],
            estimator.fixed_receptacle_registered_model_height_m,
        )

        def masked_world_points(mask):
            rows, columns = np.nonzero(mask)
            pixels = np.column_stack((columns, rows)).astype(np.float32)
            values = depth[rows, columns]
            valid = np.isfinite(values) & (values > 0.0)
            return np.asarray(
                camera.get_world_points_from_image_coords(
                    pixels[valid], values[valid]
                ),
                dtype=np.float64,
            )

        # These registered point clouds are the only geometric inputs to the
        # optional Pose5D estimator below.  They are captured before any
        # simulation truth pose is read for post-hoc scoring.
        loose_world_points = masked_world_points(loose_mask)
        fixed_world_points = masked_world_points(fixed_mask)
        loose_visible_surface_xy = robust_world_xy_centroid(loose_world_points)
        fixed_visible_surface_xy = robust_world_xy_centroid(fixed_world_points)
        if pose5d_config is not None:
            from kcg_connector.d38999_pose5d import (
                estimate_pose5d,
                relative_pose5d,
            )

            if not pose5d_capture_id:
                raise ValueError("pose5d_capture_id is required")
            if pose5d_axis_priors is None or set(pose5d_axis_priors) != {
                "loose_plug",
                "fixed_receptacle",
            }:
                raise ValueError("Pose5D requires both endpoint axis priors")
            if pose5d_authorization_gates is None:
                raise ValueError("Pose5D authorization gates are required")
            lateral_gate = float(
                pose5d_authorization_gates["lateral_position_m"]
            )
            axis_gate = float(pose5d_authorization_gates["axis_angle_rad"])
            capture_timestamp_iso = datetime.now(timezone.utc).isoformat()

            def pose5d_endpoint(label, points, statistics):
                valid_ratio = (
                    float(statistics.valid_depth_count)
                    / float(max(1, statistics.pixel_count))
                )
                return estimate_pose5d(
                    points,
                    object_id=label,
                    frame_id="world",
                    capture_id=pose5d_capture_id,
                    axis_prior=pose5d_axis_priors[label],
                    depth_valid_ratio=valid_ratio,
                    lateral_authorization_gate_m=lateral_gate,
                    axis_authorization_gate_rad=axis_gate,
                    timestamp=capture_timestamp_iso,
                    config=pose5d_config,
                )

            loose_pose5d = pose5d_endpoint(
                "loose_plug", loose_world_points, loose_statistics
            )
            fixed_pose5d = pose5d_endpoint(
                "fixed_receptacle", fixed_world_points, fixed_statistics
            )
            metrics["pose5d"] = {
                "estimator_inputs": [
                    "semantic_mask",
                    "registered_depth",
                    "calibrated_or_fk_axis_prior",
                ],
                "forbidden_control_inputs": {
                    "object_truth": False,
                    "physx_contact_normal": False,
                    "collider_identity": False,
                },
                "authorization_gates": {
                    "lateral_position_m": lateral_gate,
                    "axis_angle_rad": axis_gate,
                },
                "loose_plug": loose_pose5d.to_dict(),
                "fixed_receptacle": fixed_pose5d.to_dict(),
                "relative_receptacle_plug": relative_pose5d(
                    loose_pose5d, fixed_pose5d
                ),
            }
        loose_position_raw, loose_orientation = body.get_world_pose()
        loose_position_array = np.asarray(
            loose_position_raw, dtype=np.float64
        )
        fixed_position_raw, fixed_orientation = world_pose(
            bindings, fixed_prim
        )
        fixed_position_array = np.asarray(
            fixed_position_raw, dtype=np.float64
        )
        if (
            loose_position_array.shape != (3,)
            or fixed_position_array.shape != (3,)
            or not np.all(np.isfinite(loose_position_array))
            or not np.all(np.isfinite(fixed_position_array))
        ):
            raise RuntimeError("endpoint truth poses must be finite")
        loose_position = tuple(float(value) for value in loose_position_array)
        fixed_position = tuple(float(value) for value in fixed_position_array)
        loose_orientation = tuple(
            float(value) for value in loose_orientation
        )
        fixed_orientation = tuple(
            float(value) for value in fixed_orientation
        )
        loose_xy_error = float(
            np.linalg.norm(
                np.asarray(loose_estimated_world[:2])
                - loose_position_array[:2]
            )
        )
        fixed_xy_error = float(
            np.linalg.norm(
                np.asarray(fixed_estimated_world[:2])
                - fixed_position_array[:2]
            )
        )
        loose_visible_xy_error = float(
            np.linalg.norm(
                np.asarray(loose_visible_surface_xy)
                - loose_position_array[:2]
            )
        )
        fixed_visible_xy_error = float(
            np.linalg.norm(
                np.asarray(fixed_visible_surface_xy)
                - fixed_position_array[:2]
            )
        )
        acceptance = rgbd.acceptance

        def endpoint_gate(statistics, error, truth_frame, mask_frame):
            return bool(
                truth_frame
                and mask_frame
                and statistics.pixel_count
                >= acceptance.minimum_pixels_per_endpoint
                and statistics.visible_fraction
                >= acceptance.minimum_visible_fraction_per_endpoint
                and statistics.minimum_depth_m
                >= acceptance.minimum_valid_depth_m
                and statistics.maximum_depth_m
                <= acceptance.maximum_valid_depth_m
                and error <= acceptance.maximum_xy_centroid_error_m
            )

        loose_gate = endpoint_gate(
            loose_statistics,
            loose_xy_error,
            projection_records["loose_plug"]["in_frame"],
            mask_center_records["loose_plug"]["in_frame"],
        )
        fixed_gate = endpoint_gate(
            fixed_statistics,
            fixed_xy_error,
            projection_records["fixed_receptacle"]["in_frame"],
            mask_center_records["fixed_receptacle"]["in_frame"],
        )
        metrics.update(
            {
                "camera": {
                    "clipping_range_m": list(camera.get_clipping_range()),
                    "focal_length_mm": camera.get_focal_length(),
                    "frame_id": rgbd.camera.frame_id,
                    "horizontal_aperture_mm": (
                        camera.get_horizontal_aperture()
                    ),
                    "intrinsics": _host_numpy(
                        camera.get_intrinsics_matrix(), dtype=np.float64
                    ).tolist(),
                    "prim_path": rgbd.camera.prim_path,
                    "resolution": list(rgbd.camera.resolution),
                },
                "camera_observation_present": bool(loose_gate and fixed_gate),
                "fixed_receptacle": {
                    "mask_depth": asdict(fixed_statistics),
                    "passed": fixed_gate,
                    "ray_plane_registered_model_height_world_xyz_m": list(
                        fixed_estimated_world
                    ),
                    "registered_model_height_m": (
                        estimator.fixed_receptacle_registered_model_height_m
                    ),
                    "registered_model_height_source": (
                        estimator.
                        fixed_receptacle_registered_model_height_source
                    ),
                    "registered_truth_xy_m": fixed_position_array[:2].tolist(),
                    "semantic_mask_center": (
                        mask_center_records["fixed_receptacle"]
                    ),
                    "visible_surface_depth_median_world_xy_m": list(
                        fixed_visible_surface_xy
                    ),
                    "visible_surface_depth_median_xy_error_m": (
                        fixed_visible_xy_error
                    ),
                    "xy_error_m": fixed_xy_error,
                },
                "loose_plug": {
                    "mask_depth": asdict(loose_statistics),
                    "passed": loose_gate,
                    "ray_plane_registered_model_height_world_xyz_m": list(
                        loose_estimated_world
                    ),
                    "registered_model_height_m": (
                        estimator.loose_plug_registered_model_height_m
                    ),
                    "registered_model_height_source": (
                        estimator.loose_plug_registered_model_height_source
                    ),
                    "registered_truth_xy_m": (
                        loose_position_array[:2].tolist()
                    ),
                    "semantic_mask_center": (
                        mask_center_records["loose_plug"]
                    ),
                    "visible_surface_depth_median_world_xy_m": list(
                        loose_visible_surface_xy
                    ),
                    "visible_surface_depth_median_xy_error_m": (
                        loose_visible_xy_error
                    ),
                    "xy_error_m": loose_xy_error,
                },
                "position_estimator": {
                    "depth_mask_role": (
                        "visibility_and_visible_surface_diagnostic_only"
                    ),
                    "kind": estimator.kind,
                    "mask_center_statistic": estimator.mask_center_statistic,
                    "ray_parallel_gate_passed": True,
                    "uses_registered_truth_xy": False,
                },
                "rgbd_position_estimate_scope": (
                    "ray_plane_registered_model_height_world_xy_only"
                ),
            }
        )
        if output_dir is not None:
            _save_capture_artifacts(
                bindings=bindings,
                rgbd=rgbd,
                output_dir=output_dir,
                rgba=rgba,
                depth=depth,
                semantic=semantic,
                loose_mask=loose_mask,
                fixed_mask=fixed_mask,
            )
            metrics["output_directory"] = str(Path(output_dir))
            if "pose5d" in metrics:
                from PIL import ImageDraw

                overlay = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
                overlay[loose_mask] = (
                    0.45 * overlay[loose_mask]
                    + 0.55 * np.asarray((235, 80, 70))
                ).astype(np.uint8)
                overlay[fixed_mask] = (
                    0.45 * overlay[fixed_mask]
                    + 0.55 * np.asarray((70, 180, 245))
                ).astype(np.uint8)
                image = bindings["Image"].fromarray(overlay)
                draw = ImageDraw.Draw(image)
                colors = {
                    "loose_plug": (255, 215, 40),
                    "fixed_receptacle": (40, 255, 170),
                }
                for row, role in enumerate(
                    ("loose_plug", "fixed_receptacle")
                ):
                    pose = metrics["pose5d"][role]
                    center = np.asarray(pose["xyz_m"], dtype=np.float64)
                    axis = np.asarray(pose["axis_vector"], dtype=np.float64)
                    pixels = np.asarray(
                        camera.get_image_coords_from_world_points(
                            np.vstack((center, center + 0.060 * axis))
                        ),
                        dtype=np.float64,
                    )
                    if pixels.shape == (2, 2) and np.all(np.isfinite(pixels)):
                        start = tuple(float(value) for value in pixels[0])
                        end = tuple(float(value) for value in pixels[1])
                        color = colors[role]
                        draw.ellipse(
                            (
                                start[0] - 5,
                                start[1] - 5,
                                start[0] + 5,
                                start[1] + 5,
                            ),
                            fill=color,
                            outline=(0, 0, 0),
                        )
                        draw.line((start, end), fill=color, width=4)
                    yaw_values = [
                        item["yaw_rad"]
                        for item in pose["yaw_hypotheses"]
                    ]
                    draw.text(
                        (12, 12 + row * 56),
                        (
                            f"{role}: roll={pose['roll_rad']:.4f} "
                            f"pitch={pose['pitch_rad']:.4f} "
                            f"C2=({yaw_values[0]:.3f},{yaw_values[1]:.3f}) "
                            f"conf={pose['confidence']:.3f} "
                            f"authorized={pose['control_authorized']}"
                        ),
                        fill=colors[role],
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )
                    if pose["reject_reason"]:
                        draw.text(
                            (12, 34 + row * 56),
                            str(pose["reject_reason"]),
                            fill=(255, 120, 100),
                            stroke_width=2,
                            stroke_fill=(0, 0, 0),
                        )
                image.save(Path(output_dir) / "pose5d_overlay.png")
        metrics["passed"] = metrics["camera_observation_present"] is True
    except Exception as exception:
        metrics.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "passed": False,
            }
        )
        if rgba is not None and "Image" in bindings:
            diagnostic_path = Path(
                "/tmp/kcg_d38999_rgbd_failed_rgb.png"
            )
            bindings["Image"].fromarray(
                np.asarray(rgba[:, :, :3], dtype=np.uint8)
            ).save(diagnostic_path)
            metrics["failed_rgb_diagnostic"] = str(diagnostic_path)
    finally:
        cleanup = cleanup_rgbd_runtime_resources(
            annotators, camera, render_product
        )
        metrics["resource_cleanup"] = cleanup
        if cleanup["resources_released"] is not True:
            metrics["passed"] = False
        timeline_state = restore_timeline_after_rgbd_capture(
            world,
            simulation_app,
            was_playing=timeline_playing_before,
        )
        metrics["timeline_state"] = timeline_state
        if timeline_pause is None:
            metrics["passed"] = False
        if timeline_state["restored"] is not True:
            metrics["passed"] = False
    return D38999RgbdRuntimeCapture(
        metrics=metrics,
        loose_position_world_m=loose_position,
        loose_orientation_wxyz=loose_orientation,
        fixed_position_world_m=fixed_position,
        fixed_orientation_wxyz=fixed_orientation,
    )


@dataclass(frozen=True)
class D38999RgbdRawFormalCapture:
    """Raw RGB+depth capture with no semantic annotator and no endpoint truth."""

    metrics: dict[str, Any]
    rgb: np.ndarray | None
    depth: np.ndarray | None

    @property
    def passed(self) -> bool:
        return self.metrics.get("passed") is True


def _save_raw_formal_artifacts(*, bindings, rgbd, output_dir, rgb, depth):
    Image = bindings["Image"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        output_path / rgbd.output.rgb_filename
    )
    np.save(output_path / rgbd.output.depth_numpy_filename, depth)


def capture_d38999_rgbd_raw_formal(
    *,
    bindings: Mapping[str, Any],
    simulation_app,
    world,
    stage,
    tabletop,
    rgbd,
    output_dir: Path | str | None = None,
    camera_clipping_range_m=None,
    rt_subframes: int = 4,
) -> D38999RgbdRawFormalCapture:
    """Capture only Replicator ``rgb`` and ``distance_to_image_plane``.

    This function has no semantic annotator, no endpoint labels, no endpoint
    prim arguments, and no object pose/velocity calls.  Its success can
    therefore never depend on Isaac semantic truth or object truth.
    """
    # ``ensure_d38999_rgbd_stage_prims`` validates the FK-authored camera
    # transform with ``Usd.TimeCode.Default``; keep this transitive dependency
    # explicit so a missing binding fails at the API boundary, not mid-capture.
    required = {"Camera", "Gf", "Usd", "UsdGeom", "UsdLux", "rep"}
    if output_dir is not None:
        required.add("Image")
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"missing raw RGB-D runtime bindings: {missing}")

    Camera = bindings["Camera"]
    rep = bindings["rep"]

    metrics: dict[str, Any] = {
        "capture_kind": "raw_rgbd_formal_only",
        "semantic_annotator_used": False,
        "endpoint_semantic_read": False,
        "endpoint_truth_read": False,
        "object_pose_calls": 0,
        "passed": False,
    }
    camera = None
    render_product = None
    annotators = []
    rgba = None
    depth = None
    timeline_playing_before = bool(world.is_playing())
    timeline_pause = None
    try:
        timeline_pause = pause_timeline_for_rgbd_capture(
            world, simulation_app, was_playing=timeline_playing_before
        )
        metrics["timeline_pause"] = timeline_pause
        if timeline_pause["paused_for_capture"] is not True:
            raise RuntimeError("raw RGB-D capture could not pause the timeline")
        camera_prim, stage_prim_lifecycle = ensure_d38999_rgbd_stage_prims(
            bindings=bindings,
            stage=stage,
            tabletop=tabletop,
            rgbd=rgbd,
            camera_clipping_range_m=camera_clipping_range_m,
        )
        metrics["stage_prim_lifecycle"] = stage_prim_lifecycle
        render_product = rep.create.render_product(
            camera_prim, rgbd.camera.resolution, name="D38999RawFormalRenderProduct"
        )
        render_product_path = render_product.path
        camera = Camera(
            prim_path=rgbd.camera.prim_path,
            name="d38999_raw_formal_camera",
            frequency=rgbd.camera.frequency_hz,
            resolution=rgbd.camera.resolution,
            render_product_path=render_product_path,
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        annotators.extend((rgb_annotator, depth_annotator))
        for annotator in annotators:
            annotator.attach([render_product_path])
        metrics["render_pipeline"] = {
            "annotator_api": "rep.AnnotatorRegistry.get_annotator",
            "annotators": ["rgb", "distance_to_image_plane"],
            "semantic_segmentation_requested": False,
            "render_product_path": render_product_path,
        }
        if not isinstance(rt_subframes, int) or rt_subframes < 1:
            raise ValueError("rt_subframes must be a positive integer")
        metrics["render_pipeline"]["rt_subframes"] = int(rt_subframes)
        render_start_s = __import__("time").perf_counter()
        for _ in range(rgbd.camera.warmup_frames):
            rep.orchestrator.step(
                rt_subframes=int(rt_subframes),
                delta_time=0.0,
                pause_timeline=True,
            )
        metrics["render_pipeline"]["capture_render_seconds"] = float(
            __import__("time").perf_counter() - render_start_s
        )
        rgba = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
        metrics["capture_timestamp_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        if rgba.ndim not in (2, 3):
            raise RuntimeError("raw RGB output must be 2D or 3D")
        if rgba.ndim == 3 and rgba.shape[2] < 3:
            raise RuntimeError("raw RGB output must have at least 3 channels")
        if rgba.shape[:2] != depth.shape:
            raise RuntimeError("raw RGB and depth shapes differ")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if rgb.shape[:2] != depth.shape:
            raise RuntimeError("raw RGB/depth shape mismatch")
        finite_positive_depth_pixels = int(np.sum(valid_depth))
        metrics["camera_frame_diagnostics"] = {
            "depth_valid_fraction": float(np.mean(valid_depth)),
            "finite_positive_depth_pixels": finite_positive_depth_pixels,
            "minimum_finite_positive_depth_pixels_candidate": 1,
            "inf_depth_pixels_preserved": int(
                np.sum(np.isinf(depth))
            ),
            "rgb_maximum": int(np.max(rgb)),
            "rgb_minimum": int(np.min(rgb)),
            "rgb_mean": float(np.mean(rgb)),
            "rgb_dtype": str(rgb.dtype),
        }
        metrics["camera"] = {
            "clipping_range_m": list(camera.get_clipping_range()),
            "focal_length_mm": camera.get_focal_length(),
            "frame_id": rgbd.camera.frame_id,
            "horizontal_aperture_mm": camera.get_horizontal_aperture(),
            "intrinsics": _host_numpy(
                camera.get_intrinsics_matrix(), dtype=np.float64
            ).tolist(),
            "prim_path": rgbd.camera.prim_path,
            "resolution": list(rgbd.camera.resolution),
        }
        if output_dir is not None:
            _save_raw_formal_artifacts(
                bindings=bindings,
                rgbd=rgbd,
                output_dir=output_dir,
                rgb=rgb,
                depth=depth,
            )
            metrics["output_directory"] = str(Path(output_dir))
        metrics["passed"] = bool(
            depth.shape == rgb.shape[:2]
            and depth.size > 0
            and np.all(np.isfinite(rgb))
            and finite_positive_depth_pixels >= 1
        )
    finally:
        cleanup = cleanup_rgbd_runtime_resources(
            annotators, camera, render_product
        )
        metrics["resource_cleanup"] = cleanup
        timeline_state = restore_timeline_after_rgbd_capture(
            world, simulation_app, was_playing=timeline_playing_before
        )
        metrics["timeline_state"] = timeline_state
        if timeline_pause is None or timeline_state["restored"] is not True:
            metrics["passed"] = False
        if cleanup["resources_released"] is not True:
            metrics["passed"] = False
    return D38999RgbdRawFormalCapture(
        metrics=metrics,
        rgb=rgb,
        depth=depth,
    )


__all__ = [
    "D38999RgbdRawFormalCapture",
    "D38999RgbdRuntimeCapture",
    "capture_d38999_rgbd_raw_formal",
    "capture_d38999_rgbd_runtime",
    "endpoint_projection_records",
    "ensure_d38999_rgbd_stage_prims",
    "pause_timeline_for_rgbd_capture",
    "restore_timeline_after_rgbd_capture",
]
