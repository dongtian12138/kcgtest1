#!/usr/bin/env python3

"""Isolated Isaac palm-front RGB-D probe for the public-spec keyed-v2 plug.

This is a bounded visual probe, not an insertion runner.  It opens the new
keyed-v2 asset directly, removes the fixed receptacle from the in-memory stage,
and renders the loose plug with a fixed axial camera.  Only ``rgb`` and
``distance_to_image_plane`` are requested.  The face and occlusion masks are
constructed from the rendered depth under an explicit *isolated-probe-only*
assumption, then passed to the CPU key shadow pipeline.

No semantic annotation, object-pose query, contact report, or collider query is
part of this file.  A successful result remains shadow-only and can never
authorize control.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np


SCHEMA_VERSION = "kcg_d38999_keyed_v2_palm_front_probe_v1"
PROBE_SCOPE = "ISOLATED_KEYED_V2_PALM_FRONT_PROBE_ONLY"
KEYED_MODEL_ID = "d38999_26kj61sn_keyed_proxy_v2"
ASSET_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/"
    "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/d38999_keyed_v2_palm_front_probe_v1"
)
ROOT_PRIM_PATH = "/World/D38999Shell25JKeyedPublicSpecV2"
FIXED_RECEPTACLE_PRIM_PATH = ROOT_PRIM_PATH + "/FixedReceptacle"
LOOSE_PLUG_PRIM_PATH = ROOT_PRIM_PATH + "/LoosePlug"
CAMERA_PRIM_PATH = "/World/KeyedV2PalmFrontProbeCamera"
LIGHT_PRIM_PATH = "/World/KeyedV2PalmFrontProbeDomeLight"
CAMERA_RESOLUTION = (640, 640)
CAMERA_EYE_WORLD_M = (0.0, 0.0, 0.060)
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_APERTURE_MM = 20.955
CAMERA_CLIPPING_RANGE_M = (0.02, 1.0)
FRONT_SURFACE_BAND_M = 0.0015
MINIMUM_FACE_PIXELS = 200
BRANCH_DIRECTIONS_UV = ((1.0, 0.0), (-1.0, 0.0))


def _arguments(repository: Path, argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Render the isolated keyed-v2 plug face and run key shadow "
            "selection without semantic or object truth"
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(repository / DEFAULT_OUTPUT_RELATIVE_PATH),
        help="new evidence directory; existing paths are never overwritten",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--rt-subframes", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.warmup_frames < 1:
        parser.error("--warmup-frames must be positive")
    if arguments.rt_subframes < 1:
        parser.error("--rt-subframes must be positive")
    return arguments


def resolve_new_output_directory(path: Path | str) -> Path:
    """Resolve an output directory while refusing every existing target."""
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    if output.name in {"", ".", ".."}:
        raise ValueError("probe output directory must have a concrete name")
    return output


def _center_component(mask: np.ndarray) -> np.ndarray:
    """Return the component nearest image centre using only image pixels."""
    height, width = mask.shape
    candidates = np.argwhere(mask)
    if candidates.size == 0:
        return np.zeros_like(mask)
    image_center = np.asarray(
        ((height - 1) / 2.0, (width - 1) / 2.0), dtype=np.float64
    )
    distances = np.sum((candidates.astype(np.float64) - image_center) ** 2, axis=1)
    seed_v, seed_u = (int(value) for value in candidates[int(np.argmin(distances))])

    component = np.zeros_like(mask)
    component[seed_v, seed_u] = True
    stack = [(seed_v, seed_u)]
    while stack:
        v, u = stack.pop()
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dv == 0 and du == 0:
                    continue
                nv, nu = v + dv, u + du
                if (
                    0 <= nv < height
                    and 0 <= nu < width
                    and mask[nv, nu]
                    and not component[nv, nu]
                ):
                    component[nv, nu] = True
                    stack.append((nv, nu))
    return component


def derive_isolated_probe_inputs(
    depth_m: Any,
    *,
    front_surface_band_m: float = FRONT_SURFACE_BAND_M,
) -> dict[str, Any]:
    """Derive detector inputs from depth for this isolated probe only.

    The nearest connected surface is treated as the plug mating face.  The
    all-false occlusion mask is *not* a general occlusion estimator: it is valid
    only because this probe disables the other endpoint and authors no hand,
    table, or other occluder.
    """
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth_m must be a numeric image with shape (H, W)")
    depth = depth.astype(np.float64, copy=False)
    if (
        isinstance(front_surface_band_m, bool)
        or not math.isfinite(float(front_surface_band_m))
        or float(front_surface_band_m) <= 0.0
    ):
        raise ValueError("front_surface_band_m must be finite and positive")

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_values = depth[valid_depth]
    if valid_values.size < MINIMUM_FACE_PIXELS:
        raise ValueError("isolated RGB-D frame has insufficient valid depth")
    front_reference_m = float(np.quantile(valid_values, 0.02))
    front_limit_m = front_reference_m + float(front_surface_band_m)
    front_candidates = valid_depth & (depth <= front_limit_m)
    face = _center_component(front_candidates)
    face_pixels = int(np.count_nonzero(face))
    if face_pixels < MINIMUM_FACE_PIXELS:
        raise ValueError("nearest front-surface component is too small")

    rows, columns = np.nonzero(face)
    face_center_uv = (
        float(np.mean(columns)),
        float(np.mean(rows)),
    )
    # Constructed from the observed image shape, with the value justified only
    # by the stage isolation recorded below.  This must not migrate unchanged
    # into a hand/table/integrated scene.
    occlusion = np.zeros_like(valid_depth, dtype=np.bool_)
    return {
        "connector_face_mask": face,
        "occlusion_mask": occlusion,
        "face_center_uv": face_center_uv,
        "diagnostics": {
            "scope": PROBE_SCOPE,
            "face_mask_source": (
                "DISTANCE_TO_IMAGE_PLANE_NEAREST_CONNECTED_FRONT_SURFACE_BAND"
            ),
            "front_depth_reference_m": front_reference_m,
            "front_surface_band_m": float(front_surface_band_m),
            "front_depth_limit_m": front_limit_m,
            "valid_depth_pixels": int(valid_values.size),
            "face_pixels": face_pixels,
            "face_center_source": "DEPTH_MASK_PIXEL_CENTROID_UV",
            "occlusion_mask_source": (
                "ALL_FALSE_FROM_EXPLICIT_OTHER_ENDPOINT_AND_OCCLUDER_ISOLATION"
            ),
            "occlusion_estimator_general_scene_valid": False,
            "integrated_runtime_input_claimed": False,
        },
    }


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return preview
    low, high = np.quantile(depth[valid], (0.02, 0.98))
    if high <= low:
        preview[valid] = 255
        return preview
    normalized = np.clip((high - depth[valid]) / (high - low), 0.0, 1.0)
    preview[valid] = np.round(255.0 * normalized).astype(np.uint8)
    return preview


def _json_shadow_result(result: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    document = copy.deepcopy(result)
    detector = document.get("key_region_detection")
    if not isinstance(detector, dict) or "key_probability" not in detector:
        raise RuntimeError("shadow result is missing key_probability")
    probability = np.asarray(detector.pop("key_probability"), dtype=np.float64)
    detector["key_probability_artifact"] = {
        "path": "key_probability.npy",
        "shape": list(probability.shape),
        "minimum": float(np.min(probability)),
        "maximum": float(np.max(probability)),
        "nonzero_pixels": int(np.count_nonzero(probability)),
    }
    return document, probability


def _require_shadow_only(result: dict[str, Any]) -> None:
    if result.get("control_authorized") is not False:
        raise RuntimeError("probe result attempted to authorize control")
    if result.get("selected_for_control_allowed") is not False:
        raise RuntimeError("probe result relaxed selected-for-control boundary")
    detector = result.get("key_region_detection")
    if not isinstance(detector, dict) or detector.get("control_authorized") is not False:
        raise RuntimeError("key detector control boundary is missing")
    selector = result.get("key_branch_selection")
    if selector is not None and selector.get("control_authorized") is not False:
        raise RuntimeError("key selector control boundary is missing")


def _capture_isolated_rgbd(
    *,
    simulation_app,
    asset_path: Path,
    warmup_frames: int,
    rt_subframes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Open the isolated stage and request exactly RGB plus planar depth."""
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    context = omni.usd.get_context()
    if context.open_stage(str(asset_path)) is not True:
        raise RuntimeError(f"could not open keyed-v2 stage: {asset_path}")
    for _ in range(3):
        simulation_app.update()
    stage = context.get_stage()
    root_prim = stage.GetPrimAtPath(ROOT_PRIM_PATH)
    loose_prim = stage.GetPrimAtPath(LOOSE_PLUG_PRIM_PATH)
    fixed_prim = stage.GetPrimAtPath(FIXED_RECEPTACLE_PRIM_PATH)
    for label, prim in (
        ("keyed-v2 root", root_prim),
        ("loose plug", loose_prim),
        ("fixed receptacle", fixed_prim),
    ):
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"{label} prim is missing")

    UsdGeom.Imageable(fixed_prim).MakeInvisible()
    fixed_prim.SetActive(False)

    camera = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*CAMERA_EYE_WORLD_M))
    camera_xform.AddTransformOp().Set(matrix)
    camera.CreateFocalLengthAttr(CAMERA_FOCAL_LENGTH_MM)
    camera.CreateHorizontalApertureAttr(CAMERA_APERTURE_MM)
    camera.CreateVerticalApertureAttr(CAMERA_APERTURE_MM)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*CAMERA_CLIPPING_RANGE_M))

    light = UsdLux.DomeLight.Define(stage, LIGHT_PRIM_PATH)
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    render_product = None
    annotators = []
    try:
        render_product = rep.create.render_product(
            camera.GetPrim(),
            CAMERA_RESOLUTION,
            name="D38999KeyedV2PalmFrontProbeProduct",
        )
        render_product_path = render_product.path
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        annotators.extend((rgb_annotator, depth_annotator))
        for annotator in annotators:
            annotator.attach([render_product_path])
        for _ in range(warmup_frames):
            rep.orchestrator.step(
                rt_subframes=rt_subframes,
                delta_time=0.0,
                pause_timeline=True,
            )
        rgba = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float64)
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError("RGB annotator did not return at least three channels")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
            raise RuntimeError("RGB and distance_to_image_plane shapes differ")
        diagnostics = {
            "asset": str(asset_path),
            "stage_root_prim": ROOT_PRIM_PATH,
            "loose_plug_prim": LOOSE_PLUG_PRIM_PATH,
            "fixed_receptacle_isolated": True,
            "fixed_receptacle_visibility": "INVISIBLE_AND_INACTIVE",
            "camera_prim": CAMERA_PRIM_PATH,
            "camera_view": "FIXED_WORLD_FRONT_VIEW_ALONG_MINUS_Z",
            "camera_eye_world_m": list(CAMERA_EYE_WORLD_M),
            "camera_resolution": list(CAMERA_RESOLUTION),
            "annotators": ["rgb", "distance_to_image_plane"],
            "warmup_frames": int(warmup_frames),
            "rt_subframes": int(rt_subframes),
            "valid_depth_pixels": int(
                np.count_nonzero(np.isfinite(depth) & (depth > 0.0))
            ),
            "semantic_annotator_used": False,
            "object_pose_queries": 0,
            "contact_queries": 0,
            "collider_queries": 0,
        }
        return rgb, depth, diagnostics
    finally:
        if render_product is not None:
            render_product_path = render_product.path
            for annotator in annotators:
                try:
                    annotator.detach([render_product_path])
                except Exception:
                    pass
            try:
                render_product.destroy()
            except Exception:
                pass


def _save_artifacts(
    *,
    output_dir: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    inputs: dict[str, Any],
    result: dict[str, Any],
    capture_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    face = np.asarray(inputs["connector_face_mask"], dtype=np.bool_)
    occlusion = np.asarray(inputs["occlusion_mask"], dtype=np.bool_)
    json_result, probability = _json_shadow_result(result)

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(output_dir / "rgb.png")
    np.save(output_dir / "depth_m.npy", np.asarray(depth, dtype=np.float32))
    Image.fromarray(_depth_preview(depth)).save(output_dir / "depth_preview.png")
    np.save(output_dir / "connector_face_mask.npy", face)
    Image.fromarray(face.astype(np.uint8) * 255).save(
        output_dir / "connector_face_mask.png"
    )
    np.save(output_dir / "occlusion_mask.npy", occlusion)
    Image.fromarray(occlusion.astype(np.uint8) * 255).save(
        output_dir / "occlusion_mask.png"
    )
    np.save(output_dir / "key_probability.npy", probability)
    Image.fromarray(np.round(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
        output_dir / "key_probability.png"
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "probe_scope": PROBE_SCOPE,
        "status": json_result.get("status"),
        "passed": json_result.get("passed") is True,
        "shadow_only": True,
        "control_authorized": False,
        "selected_for_control_allowed": False,
        "keyed_model_id": KEYED_MODEL_ID,
        "capture": capture_diagnostics,
        "input_derivation": inputs["diagnostics"],
        "branch_directions_uv": [list(value) for value in BRANCH_DIRECTIONS_UV],
        "branch_direction_source": "FIXED_PROBE_IMAGE_AXIS_C2_HYPOTHESES",
        "shadow_result": json_result,
        "artifacts": {
            "rgb": "rgb.png",
            "depth_m": "depth_m.npy",
            "depth_preview": "depth_preview.png",
            "connector_face_mask": "connector_face_mask.npy",
            "connector_face_mask_preview": "connector_face_mask.png",
            "occlusion_mask": "occlusion_mask.npy",
            "occlusion_mask_preview": "occlusion_mask.png",
            "key_probability": "key_probability.npy",
            "key_probability_preview": "key_probability.png",
        },
        "claims": {
            "isolated_probe_only": True,
            "integrated_runtime_validated": False,
            "insertion_control_validated": False,
            "real_hardware_validated": False,
        },
    }
    (output_dir / "shadow_result.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> int:
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository, argv)
    asset_path = (repository / ASSET_RELATIVE_PATH).resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"keyed-v2 asset is missing: {asset_path}")
    output_dir = resolve_new_output_directory(arguments.output_dir)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    try:
        from kcg_connector.d38999_key_shadow_pipeline import (
            run_palm_key_shadow_pipeline,
        )

        rgb, depth, capture_diagnostics = _capture_isolated_rgbd(
            simulation_app=simulation_app,
            asset_path=asset_path,
            warmup_frames=arguments.warmup_frames,
            rt_subframes=arguments.rt_subframes,
        )
        inputs = derive_isolated_probe_inputs(depth)
        result = run_palm_key_shadow_pipeline(
            inputs["connector_face_mask"],
            depth,
            inputs["face_center_uv"],
            BRANCH_DIRECTIONS_UV,
            KEYED_MODEL_ID,
            occlusion_mask=inputs["occlusion_mask"],
        )
        _require_shadow_only(result)
        report = _save_artifacts(
            output_dir=output_dir,
            rgb=rgb,
            depth=depth,
            inputs=inputs,
            result=result,
            capture_diagnostics=capture_diagnostics,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "shadow_only": True,
                    "control_authorized": False,
                    "output_dir": str(output_dir),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if report["passed"] else 2
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
