#!/usr/bin/env python3

"""Render bounded review views of the frozen keyed physical-r8 asset.

The renderer opens the authorized connector USD read-only, never starts the
timeline, and never saves the in-memory camera, lights, or visibility edits.
It produces visual-review evidence only; it does not authorize a physical
bench, robot control, insertion, hardware use, randomization, or RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "kcg_d38999_keyed_physical_r8_render_review_v1"
ROOT = "/World/D38999Shell25JKeyedPhysicalV3"
FIXED = ROOT + "/FixedReceptacle"
LOOSE = ROOT + "/LoosePlug"
COUPLING_NUT = LOOSE + "/CouplingNut"
BODY_ASSEMBLY = LOOSE + "/BodyAssembly"
POLARIZING_KEYS = BODY_ASSEMBLY + "/PolarizingKeys"
CAMERA = "/World/PhysicalR8ReviewCamera"
RESOLUTION = (800, 800)
DEFAULT_ASSET = Path(
    "artifacts/kcg_connector/isaac/keyed_v3_physical_r8/"
    "d38999_shell25j_25_61_n_keyed_physical_v3_r8.usda"
)
DEFAULT_OUTPUT = Path(
    "artifacts/kcg_connector/isaac/keyed_v3_physical_r8/render_review_v1"
)
VIEWS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "plug_mating_front",
        "show": "loose",
        "eye_m": (0.0, 0.0, -0.075),
        "target_m": (0.0, 0.0, 0.010),
    },
    {
        "name": "plug_mating_oblique",
        "show": "loose",
        "eye_m": (0.065, 0.050, -0.045),
        "target_m": (0.0, 0.0, 0.012),
    },
    {
        "name": "plug_rear_oblique",
        "show": "loose",
        "eye_m": (0.065, 0.050, 0.095),
        "target_m": (0.0, 0.0, 0.015),
    },
    {
        "name": "receptacle_mating_front",
        "show": "fixed",
        "eye_m": (0.0, 0.0, 0.085),
        "target_m": (0.0, 0.0, 0.008),
    },
    {
        "name": "receptacle_mating_oblique",
        "show": "fixed",
        "eye_m": (0.060, 0.045, 0.065),
        "target_m": (0.0, 0.0, 0.008),
    },
)
KEY_DEBUG_VIEWS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "plug_five_keys_front_diagnostic",
        "show": "loose",
        "isolate_keys": True,
        "eye_m": (0.0, 0.0, -0.060),
        "target_m": (0.0, 0.0, 0.009),
    },
    {
        "name": "plug_five_keys_oblique_diagnostic",
        "show": "loose",
        "isolate_keys": False,
        "eye_m": (0.055, 0.042, -0.030),
        "target_m": (0.0, 0.0, 0.009),
    },
    {
        "name": "plug_five_keys_isolated_oblique_diagnostic",
        "show": "loose",
        "isolate_keys": True,
        "eye_m": (0.050, 0.038, -0.030),
        "target_m": (0.0, 0.0, 0.009),
    },
)


def _arguments(
    repository: Path, argv: Sequence[str] | None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=str(repository / DEFAULT_ASSET))
    parser.add_argument(
        "--output-dir", default=str(repository / DEFAULT_OUTPUT)
    )
    parser.add_argument("--warmup-frames", type=int, default=6)
    parser.add_argument("--rt-subframes", type=int, default=2)
    parser.add_argument(
        "--key-debug",
        action="store_true",
        help=(
            "render a diagnostic plug view with the coupling nut hidden and "
            "the main/minor polarizing keys recolored in memory"
        ),
    )
    result = parser.parse_args(argv)
    if result.warmup_frames < 1 or result.rt_subframes < 1:
        parser.error("warmup frames and RT subframes must be positive")
    return result


def _new_output(path: Path | str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists():
        raise FileExistsError(
            f"refusing to overwrite render evidence: {result}"
        )
    return result


def _prim(stage: Any, path: str) -> Any:
    result = stage.GetPrimAtPath(path)
    if result is None or not result.IsValid():
        raise RuntimeError(f"required prim is missing: {path}")
    return result


def _apply_key_debug_overrides(
    stage: Any, gf: Any, usd: Any, usd_geom: Any
) -> dict[str, Any]:
    usd_geom.Imageable(_prim(stage, COUPLING_NUT)).MakeInvisible()
    key_rows: list[dict[str, Any]] = []
    for index in range(5):
        key_path = f"{POLARIZING_KEYS}/Key_{index}"
        key_prim = _prim(stage, key_path)
        color = (
            gf.Vec3f(1.0, 0.25, 0.02)
            if index == 0
            else gf.Vec3f(1.0, 0.78, 0.02)
        )
        mesh_count = 0
        for descendant in usd.PrimRange(key_prim):
            if descendant.IsA(usd_geom.Mesh):
                usd_geom.Gprim(descendant).GetDisplayColorAttr().Set([color])
                mesh_count += 1
        if mesh_count < 1:
            raise RuntimeError(f"no key meshes found below {key_path}")
        key_rows.append(
            {
                "key_index": index,
                "role": "main" if index == 0 else "minor",
                "diagnostic_color_rgb": list(color),
                "mesh_count": mesh_count,
            }
        )
    return {
        "applied_in_memory_only": True,
        "asset_saved": False,
        "coupling_nut_hidden": True,
        "main_key_color": "orange",
        "minor_key_color": "yellow",
        "keys": key_rows,
    }


def _key_isolation_imageables(stage: Any, usd_geom: Any) -> tuple[Any, ...]:
    body = _prim(stage, BODY_ASSEMBLY)
    result = []
    for child in body.GetChildren():
        if child.GetPath().pathString != POLARIZING_KEYS:
            result.append(usd_geom.Imageable(child))
    if not result:
        raise RuntimeError(
            "no surrounding body groups found for key isolation"
        )
    return tuple(result)


def _capture_view(
    *,
    rep: Any,
    viewport_manager: Any,
    camera: Any,
    fixed_imageable: Any,
    loose_imageable: Any,
    key_isolation_imageables: Sequence[Any],
    view: Mapping[str, Any],
    warmup_frames: int,
    rt_subframes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if view["show"] == "loose":
        loose_imageable.MakeVisible()
        fixed_imageable.MakeInvisible()
    else:
        fixed_imageable.MakeVisible()
        loose_imageable.MakeInvisible()
    isolate_keys = bool(view.get("isolate_keys", False))
    for imageable in key_isolation_imageables:
        if isolate_keys:
            imageable.MakeInvisible()
        else:
            imageable.MakeVisible()
    viewport_manager.set_camera_view(
        camera=camera,
        eye=np.asarray(view["eye_m"], dtype=np.float64),
        target=np.asarray(view["target_m"], dtype=np.float64),
    )
    product = rep.create.render_product(
        camera.GetPrim(), RESOLUTION, name=f"PhysicalR8_{view['name']}"
    )
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    try:
        annotator.attach([product.path])
        for _ in range(warmup_frames):
            rep.orchestrator.step(
                rt_subframes=rt_subframes,
                delta_time=0.0,
                pause_timeline=True,
            )
        rgba = np.asarray(annotator.get_data())
        valid_shape = (
            rgba.ndim == 3
            and rgba.shape[:2] == RESOLUTION[::-1]
            and rgba.shape[2] >= 3
        )
        if not valid_shape:
            raise RuntimeError(
                f"invalid RGB frame for {view['name']}: {rgba.shape}"
            )
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        near_black_fraction = float(np.mean(np.max(rgb, axis=2) <= 2))
        return rgb, {
            "eye_m": list(view["eye_m"]),
            "target_m": list(view["target_m"]),
            "shown_endpoint": view["show"],
            "keys_isolated": isolate_keys,
            "shape": list(rgb.shape),
            "mean_rgb": [float(value) for value in np.mean(rgb, axis=(0, 1))],
            "std_rgb": [float(value) for value in np.std(rgb, axis=(0, 1))],
            "near_black_pixel_fraction": near_black_fraction,
            "nonblack_passed": bool(near_black_fraction < 0.95),
        }
    finally:
        try:
            annotator.detach([product.path])
        finally:
            product.destroy()


def main(argv: Sequence[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository, argv)
    asset = Path(arguments.asset).expanduser().resolve()
    output = _new_output(arguments.output_dir)
    if not asset.is_file():
        raise FileNotFoundError(asset)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "enable_crashreporter": False,
        }
    )
    try:
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.rendering_manager import ViewportManager
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux

        from kcg_connector.d38999_keyed_v2_a2_readback_result import (
            validate_a2_composed_asset_release,
        )

        release = validate_a2_composed_asset_release(asset)
        context = omni.usd.get_context()
        if context.open_stage(str(asset)) is not True:
            raise RuntimeError(
                "failed to open the authorized physical-r8 asset"
            )
        for _ in range(3):
            app.update()
        stage = context.get_stage()
        fixed_imageable = UsdGeom.Imageable(_prim(stage, FIXED))
        loose_imageable = UsdGeom.Imageable(_prim(stage, LOOSE))
        camera = UsdGeom.Camera.Define(stage, CAMERA)
        camera.CreateFocalLengthAttr(32.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 1.0))
        dome = UsdLux.DomeLight.Define(stage, "/World/PhysicalR8Fill")
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/PhysicalR8Key")
        key.CreateIntensityAttr(1700.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.90, 0.82))
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(
            Gf.Vec3f(-35.0, 25.0, 30.0)
        )

        diagnostic_overrides = None
        key_isolation_imageables: tuple[Any, ...] = ()
        views = VIEWS
        if arguments.key_debug:
            diagnostic_overrides = _apply_key_debug_overrides(
                stage, Gf, Usd, UsdGeom
            )
            key_isolation_imageables = _key_isolation_imageables(
                stage, UsdGeom
            )
            views = KEY_DEBUG_VIEWS

        output.mkdir(parents=True, exist_ok=False)
        view_reports: dict[str, Any] = {}
        for view in views:
            rgb, diagnostics = _capture_view(
                rep=rep,
                viewport_manager=ViewportManager,
                camera=camera,
                fixed_imageable=fixed_imageable,
                loose_imageable=loose_imageable,
                key_isolation_imageables=key_isolation_imageables,
                view=view,
                warmup_frames=arguments.warmup_frames,
                rt_subframes=arguments.rt_subframes,
            )
            path = output / f"{view['name']}.png"
            Image.fromarray(rgb, "RGB").save(path)
            diagnostics["file"] = path.name
            view_reports[str(view["name"])] = diagnostics

        nonblack = all(row["nonblack_passed"] for row in view_reports.values())
        report = {
            "schema_version": SCHEMA_VERSION,
            "asset": str(asset),
            "asset_revision": "keyed_v3_physical_r8",
            "key_debug_mode": bool(arguments.key_debug),
            "diagnostic_overrides": diagnostic_overrides,
            "timeline_started": False,
            "immutable_output_guard": True,
            "a2_static_release_readback": {
                "passed": bool(release.release_evidence),
                "collider_row_count": release.collider_row_count,
                "property_row_count": release.property_row_count,
                "family_pair_row_count": release.family_pair_row_count,
                "filter_source_row_count": release.filter_source_row_count,
            },
            "views": view_reports,
            "render_nonblack_passed": nonblack,
            "human_visual_review_complete": False,
            "physical_benches_passed": False,
            "control_authorized": False,
            "hardware_authorized": False,
            "randomization_authorized": False,
            "rl_authorized": False,
            "manufacturer_cad_fidelity_claimed": False,
            "passed_before_human_review": bool(
                release.release_evidence and nonblack
            ),
        }
        report_path = output / "physical_r8_render_report.json"
        report_path.write_text(
            json.dumps(
                report, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        return 0 if report["passed_before_human_review"] else 2
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
