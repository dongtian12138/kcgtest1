#!/usr/bin/env python3
"""Render faithful multiview review images from the accepted J599 USDC.

The source asset is opened read-only in practice: all cameras, lights,
visibility choices and pose overrides are authored into the anonymous USD
session layer, the timeline is never started, and the source hash is checked
before and after rendering.  The static mated view is for appearance review
only and is never presented as dynamic assembly evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = MODEL_ROOT / "generated" / "j599_25_35_pair_assembly.usdc"
DEFAULT_OUTPUT = MODEL_ROOT / "renders" / "multiview_v1"
PAIR_ROOT = "/World/J599_25_35_N_Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle_J599_20FJ35SN"
LOOSE_PATH = PAIR_ROOT + "/LoosePlug_J599_26FJ35PN"
BODY_PATH = LOOSE_PATH + "/Body"
NUT_PATH = LOOSE_PATH + "/CouplingNut"
CAMERA_PATH = "/World/J59925_35ReviewCamera"
RESOLUTION = (960, 960)

VIEWS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "01_pair_preassembly_oblique",
        "caption": "Pair - default import pose",
        "show": "both",
        "pose": "default",
        "eye_m": (0.105, 0.078, 0.050),
        "target_m": (0.0, 0.0, 0.002),
    },
    {
        "name": "02_pair_preassembly_side",
        "caption": "Pair - default side view",
        "show": "both",
        "pose": "default",
        "eye_m": (0.130, 0.0, 0.008),
        "target_m": (0.0, 0.0, 0.002),
    },
    {
        "name": "03_pair_static_mated_oblique",
        "caption": "Pair - static mated pose",
        "show": "both",
        "pose": "static_mated",
        "eye_m": (0.105, 0.078, 0.042),
        "target_m": (0.0, 0.0, -0.002),
    },
    {
        "name": "04_plug_mating_front",
        "caption": "J599/26 plug - mating front",
        "show": "plug",
        "pose": "default",
        "eye_m": (0.0, 0.0, -0.100),
        "target_m": (0.0, 0.0, 0.008),
    },
    {
        "name": "05_plug_mating_oblique",
        "caption": "J599/26 plug - mating oblique",
        "show": "plug",
        "pose": "default",
        "eye_m": (0.070, 0.055, -0.070),
        "target_m": (0.0, 0.0, 0.010),
    },
    {
        "name": "06_plug_rear_oblique",
        "caption": "J599/26 plug - rear oblique",
        "show": "plug",
        "pose": "default",
        "eye_m": (0.072, 0.055, 0.105),
        "target_m": (0.0, 0.0, 0.020),
    },
    {
        "name": "07_receptacle_mating_front",
        "caption": "J599/20 receptacle - mating front",
        "show": "fixed",
        "pose": "default",
        "eye_m": (0.0, 0.0, 0.090),
        "target_m": (0.0, 0.0, -0.006),
    },
    {
        "name": "08_receptacle_mating_oblique",
        "caption": "J599/20 receptacle - mating oblique",
        "show": "fixed",
        "pose": "default",
        "eye_m": (0.070, 0.055, 0.070),
        "target_m": (0.0, 0.0, -0.007),
    },
    {
        "name": "09_receptacle_rear_oblique",
        "caption": "J599/20 receptacle - rear oblique",
        "show": "fixed",
        "pose": "default",
        "eye_m": (0.070, 0.055, -0.100),
        "target_m": (0.0, 0.0, -0.015),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-frames", type=int, default=6)
    parser.add_argument("--rt-subframes", type=int, default=2)
    result = parser.parse_args(argv)
    if result.warmup_frames < 1 or result.rt_subframes < 1:
        parser.error("warmup frames and RT subframes must be positive")
    return result


def _new_output(path: Path) -> Path:
    result = path.expanduser().resolve()
    try:
        result.relative_to(MODEL_ROOT.resolve())
    except ValueError as error:
        raise ValueError("render output must remain inside the isolated model root") from error
    if result.exists():
        raise FileExistsError(f"refusing to overwrite render evidence: {result}")
    return result


def _prim(stage: Any, path: str) -> Any:
    result = stage.GetPrimAtPath(path)
    if result is None or not result.IsValid():
        raise RuntimeError(f"required prim is missing: {path}")
    return result


def _translate_op(xformable: Any, usd_geom: Any) -> Any:
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == usd_geom.XformOp.TypeTranslate:
            return operation
    return xformable.AddTranslateOp(opSuffix="renderReview")


def _set_view_state(
    *,
    view: Mapping[str, Any],
    fixed_imageable: Any,
    loose_imageable: Any,
    fixed_translate: Any,
    body_translate: Any,
    nut_translate: Any,
    default_translations: Mapping[str, tuple[float, float, float]],
    gf: Any,
) -> None:
    show = str(view["show"])
    if show not in {"both", "plug", "fixed"}:
        raise ValueError(f"unsupported visibility mode: {show}")
    if show in {"both", "fixed"}:
        fixed_imageable.MakeVisible()
    else:
        fixed_imageable.MakeInvisible()
    if show in {"both", "plug"}:
        loose_imageable.MakeVisible()
    else:
        loose_imageable.MakeInvisible()

    fixed = default_translations["fixed"]
    body = default_translations["body"]
    nut = default_translations["nut"]
    if view["pose"] == "static_mated":
        body = (body[0], body[1], 0.0)
        nut = (nut[0], nut[1], 0.0)
    elif view["pose"] != "default":
        raise ValueError(f"unsupported pose mode: {view['pose']}")
    fixed_translate.Set(gf.Vec3d(*fixed))
    body_translate.Set(gf.Vec3d(*body))
    nut_translate.Set(gf.Vec3d(*nut))


def _capture_view(
    *,
    rep: Any,
    viewport_manager: Any,
    camera: Any,
    view: Mapping[str, Any],
    output: Path,
    warmup_frames: int,
    rt_subframes: int,
    image_module: Any,
) -> tuple[Path, dict[str, Any]]:
    viewport_manager.set_camera_view(
        camera=camera,
        eye=np.asarray(view["eye_m"], dtype=np.float64),
        target=np.asarray(view["target_m"], dtype=np.float64),
    )
    product = rep.create.render_product(
        camera.GetPrim(), RESOLUTION, name=f"J59925_35_{view['name']}"
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
            raise RuntimeError(f"invalid RGB frame for {view['name']}: {rgba.shape}")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        near_black_fraction = float(np.mean(np.max(rgb, axis=2) <= 2))
        channel_std = [float(value) for value in np.std(rgb, axis=(0, 1))]
        nonblank = bool(near_black_fraction < 0.95 and max(channel_std) > 3.0)
        image_path = output / f"{view['name']}.png"
        image_module.fromarray(rgb, "RGB").save(image_path)
        return image_path, {
            "caption": str(view["caption"]),
            "file": image_path.name,
            "eye_m": list(view["eye_m"]),
            "target_m": list(view["target_m"]),
            "shown_endpoint": str(view["show"]),
            "pose": str(view["pose"]),
            "shape": list(rgb.shape),
            "mean_rgb": [float(value) for value in np.mean(rgb, axis=(0, 1))],
            "std_rgb": channel_std,
            "near_black_pixel_fraction": near_black_fraction,
            "nonblank_passed": nonblank,
            "sha256": _sha256(image_path),
            "size_bytes": image_path.stat().st_size,
        }
    finally:
        try:
            annotator.detach([product.path])
        finally:
            product.destroy()


def _make_contact_sheet(
    image_paths: Sequence[Path],
    views: Sequence[Mapping[str, Any]],
    output: Path,
    image_module: Any,
    image_draw_module: Any,
    image_font_module: Any,
) -> Path:
    cell_width = 500
    image_height = 450
    caption_height = 52
    cell_height = image_height + caption_height
    sheet = image_module.new("RGB", (cell_width * 3, cell_height * 3), (232, 235, 239))
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        image_font_module.truetype(str(font_path), 21)
        if font_path.is_file()
        else image_font_module.load_default()
    )
    draw = image_draw_module.Draw(sheet)
    for index, (path, view) in enumerate(zip(image_paths, views)):
        row, column = divmod(index, 3)
        image = image_module.open(path).convert("RGB")
        image.thumbnail((cell_width, image_height), image_module.Resampling.LANCZOS)
        x0 = column * cell_width + (cell_width - image.width) // 2
        y0 = row * cell_height + (image_height - image.height) // 2
        sheet.paste(image, (x0, y0))
        caption = str(view["caption"])
        box = draw.textbbox((0, 0), caption, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (column * cell_width + (cell_width - text_width) // 2, row * cell_height + image_height + 12),
            caption,
            fill=(22, 28, 34),
            font=font,
        )
    path = output / "00_contact_sheet_3x3.png"
    sheet.save(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    asset = arguments.asset.expanduser().resolve()
    output = _new_output(arguments.output_dir)
    if not asset.is_file():
        raise FileNotFoundError(asset)
    asset_hash_before = _sha256(asset)

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "enable_crashreporter": False,
            "fast_shutdown": True,
        }
    )
    report: dict[str, Any] = {}
    exit_code = 1
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.rendering_manager import ViewportManager
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, UsdGeom, UsdLux

        context = omni.usd.get_context()
        if context.open_stage(str(asset)) is not True:
            raise RuntimeError("failed to open J599 assembly asset")
        for _ in range(3):
            application.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())

        root = _prim(stage, PAIR_ROOT)
        fixed_prim = _prim(stage, FIXED_PATH)
        loose_prim = _prim(stage, LOOSE_PATH)
        body_prim = _prim(stage, BODY_PATH)
        nut_prim = _prim(stage, NUT_PATH)
        identity = {
            "plug_part_number": body_prim.GetCustomDataByKey("j599:partNumber"),
            "receptacle_part_number": fixed_prim.GetCustomDataByKey("j599:partNumber"),
            "contact_count": root.GetCustomDataByKey("j599:contactCount"),
            "polarization": root.GetCustomDataByKey("j599:polarization"),
            "hardware_authorized": root.GetCustomDataByKey("j599:hardwareAuthorized"),
            "hardware_exact_fidelity": root.GetCustomDataByKey("j599:hardwareExactFidelity"),
        }
        identity_exact = identity == {
            "plug_part_number": "J599/26FJ35PN",
            "receptacle_part_number": "J599/20FJ35SN",
            "contact_count": 128,
            "polarization": "N",
            "hardware_authorized": False,
            "hardware_exact_fidelity": False,
        }
        if not identity_exact:
            raise RuntimeError(f"unexpected identity readback: {identity}")

        fixed_imageable = UsdGeom.Imageable(fixed_prim)
        loose_imageable = UsdGeom.Imageable(loose_prim)
        fixed_translate = _translate_op(UsdGeom.Xformable(fixed_prim), UsdGeom)
        body_translate = _translate_op(UsdGeom.Xformable(body_prim), UsdGeom)
        nut_translate = _translate_op(UsdGeom.Xformable(nut_prim), UsdGeom)
        default_translations = {
            "fixed": tuple(float(value) for value in (fixed_translate.Get() or Gf.Vec3d(0.0))),
            "body": tuple(float(value) for value in body_translate.Get()),
            "nut": tuple(float(value) for value in nut_translate.Get()),
        }

        camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        camera.CreateFocalLengthAttr(32.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.004, 1.0))

        dome = UsdLux.DomeLight.Define(stage, "/World/J59925_35ReviewDome")
        dome.CreateIntensityAttr(950.0)
        dome.CreateColorAttr(Gf.Vec3f(0.93, 0.96, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/J59925_35ReviewKey")
        key.CreateIntensityAttr(1800.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.82))
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 28.0, 32.0))
        fill = UsdLux.DistantLight.Define(stage, "/World/J59925_35ReviewFill")
        fill.CreateIntensityAttr(850.0)
        fill.CreateColorAttr(Gf.Vec3f(0.78, 0.88, 1.0))
        UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(35.0, -42.0, -120.0))

        output.mkdir(parents=True, exist_ok=False)
        view_reports: dict[str, Any] = {}
        image_paths: list[Path] = []
        for view in VIEWS:
            _set_view_state(
                view=view,
                fixed_imageable=fixed_imageable,
                loose_imageable=loose_imageable,
                fixed_translate=fixed_translate,
                body_translate=body_translate,
                nut_translate=nut_translate,
                default_translations=default_translations,
                gf=Gf,
            )
            image_path, diagnostics = _capture_view(
                rep=rep,
                viewport_manager=ViewportManager,
                camera=camera,
                view=view,
                output=output,
                warmup_frames=arguments.warmup_frames,
                rt_subframes=arguments.rt_subframes,
                image_module=Image,
            )
            image_paths.append(image_path)
            view_reports[str(view["name"])] = diagnostics

        sheet_path = _make_contact_sheet(
            image_paths,
            VIEWS,
            output,
            Image,
            ImageDraw,
            ImageFont,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline_started = bool(timeline.is_playing())
        asset_hash_after = _sha256(asset)
        all_nonblank = all(row["nonblank_passed"] for row in view_reports.values())
        source_unchanged = asset_hash_after == asset_hash_before
        passed = bool(identity_exact and all_nonblank and source_unchanged and not timeline_started)
        report = {
            "schema_version": "j599_25_35_multiview_render_report_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": "J599-25-35-MULTIVIEW-VISUAL-REVIEW",
            "status": "STATIC_PASS" if passed else "STATIC_FAIL",
            "passed": passed,
            "source_asset": str(asset),
            "source_asset_sha256_before": asset_hash_before,
            "source_asset_sha256_after": asset_hash_after,
            "source_asset_unchanged": source_unchanged,
            "session_layer_only_overrides": True,
            "asset_saved": False,
            "timeline_started": timeline_started,
            "identity_readback": identity,
            "identity_readback_exact": identity_exact,
            "resolution": list(RESOLUTION),
            "view_count": len(view_reports),
            "views": view_reports,
            "all_views_nonblank": all_nonblank,
            "contact_sheet": {
                "file": sheet_path.name,
                "sha256": _sha256(sheet_path),
                "size_bytes": sheet_path.stat().st_size,
            },
            "human_visual_review_complete": False,
            "claims": {
                "render_is_dynamic_assembly_evidence": False,
                "static_mated_pose_is_dynamic_success_evidence": False,
                "hardware_authorized": False,
                "manufacturer_exact_fidelity": False,
            },
        }
        report_path = output / "render_report.json"
        report_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "view_count": report["view_count"],
                    "all_views_nonblank": report["all_views_nonblank"],
                    "source_asset_unchanged": report["source_asset_unchanged"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        exit_code = 0 if passed else 2
    except BaseException as error:
        traceback.print_exc()
        if output.exists():
            failure = {
                "schema_version": "j599_25_35_multiview_render_report_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": "J599-25-35-MULTIVIEW-VISUAL-REVIEW",
                "status": "ERROR",
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "source_asset_sha256_before": asset_hash_before,
                "source_asset_sha256_after": _sha256(asset),
                "source_asset_unchanged": _sha256(asset) == asset_hash_before,
                "traceback": traceback.format_exc(),
            }
            (output / "render_report.json").write_text(
                json.dumps(failure, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        exit_code = 1
    finally:
        application.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
