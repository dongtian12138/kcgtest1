#!/usr/bin/env python3

"""Render immutable checkpoint-A review views for the keyed-v2 asset.

This is a bounded visual/structural audit.  It never drives the robot, never
steps an assembly controller, and never authorizes insertion or hardware use.
Existing output directories are refused so the evidence cannot be overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


SCHEMA_VERSION = "kcg_d38999_keyed_v2_checkpoint_a_render_v1"
ASSET_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/keyed_v2_checkpoint_a_r6/"
    "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/keyed_v2_checkpoint_a_r6/render_review_v3"
)
ROOT = "/World/D38999Shell25JKeyedPublicSpecV2"
FIXED = ROOT + "/FixedReceptacle"
LOOSE = ROOT + "/LoosePlug"
BODY = LOOSE + "/BodyAssembly"
NUT = LOOSE + "/CouplingNut"
CAMERA = "/World/CheckpointAReviewCamera"
RESOLUTION = (800, 800)
VIEWS = (
    {
        "name": "plug_front",
        "show": "loose",
        "eye_m": (0.0, 0.0, 0.060),
        "target_m": (0.0, 0.0, -0.052),
    },
    {
        "name": "plug_key_oblique",
        "show": "loose",
        "eye_m": (0.070, 0.052, 0.018),
        "target_m": (0.0, 0.0, -0.058),
    },
    {
        "name": "plug_rear_oblique",
        "show": "loose",
        "eye_m": (0.075, 0.055, -0.115),
        "target_m": (0.0, 0.0, -0.073),
    },
    {
        "name": "receptacle_front",
        "show": "fixed",
        "eye_m": (0.0, 0.0, -0.080),
        "target_m": (0.0, 0.0, 0.006),
    },
)


def _arguments(repository: Path, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        default=str(repository / ASSET_RELATIVE_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repository / DEFAULT_OUTPUT_RELATIVE_PATH),
    )
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--rt-subframes", type=int, default=4)
    result = parser.parse_args(argv)
    if result.warmup_frames < 1 or result.rt_subframes < 1:
        parser.error("warmup frames and RT subframes must be positive")
    return result


def _new_output(path: Path | str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists():
        raise FileExistsError(f"refusing to overwrite render evidence: {result}")
    return result


def _prim(stage, path: str):
    result = stage.GetPrimAtPath(path)
    if result is None or not result.IsValid():
        raise RuntimeError(f"required prim is missing: {path}")
    return result


def _structural_audit(stage, UsdGeom, UsdPhysics):
    sockets = tuple(
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(BODY + "/ContactVisuals/Sockets/Socket_")
    )
    keys = tuple(
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(BODY + "/CollisionKeys/Key_")
    )
    insert_face = UsdGeom.Cylinder(
        _prim(stage, BODY + "/ContactVisuals/InsertFace")
    )
    face_center_z = float(
        insert_face.GetPrim().GetAttribute("xformOp:translate").Get()[2]
    )
    face_front_z = face_center_z + 0.5 * float(insert_face.GetHeightAttr().Get())
    socket_fronts = []
    for prim in sockets:
        cylinder = UsdGeom.Cylinder(prim)
        center_z = float(prim.GetAttribute("xformOp:translate").Get()[2])
        socket_fronts.append(center_z + 0.5 * float(cylinder.GetHeightAttr().Get()))

    collision_grip = _prim(stage, NUT + "/CollisionGripShell")
    body_mass = UsdPhysics.MassAPI(_prim(stage, BODY))
    nut_mass = UsdPhysics.MassAPI(_prim(stage, NUT))
    mass_apis = (body_mass, nut_mass)
    explicit_mass_properties = all(
        api.GetMassAttr().HasAuthoredValueOpinion()
        and api.GetCenterOfMassAttr().HasAuthoredValueOpinion()
        and api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
        and api.GetPrincipalAxesAttr().HasAuthoredValueOpinion()
        for api in mass_apis
    )
    result = {
        "socket_count": len(sockets),
        "key_count": len(keys),
        "insert_face_front_z_m": face_front_z,
        "minimum_socket_front_z_m": min(socket_fronts),
        "all_socket_fronts_ahead_of_opaque_face": all(
            value > face_front_z for value in socket_fronts
        ),
        "continuous_grip_collision_present": collision_grip.HasAPI(
            UsdPhysics.CollisionAPI
        ),
        "continuous_grip_collision_visibility": str(
            UsdGeom.Imageable(collision_grip).GetVisibilityAttr().Get()
        ),
        "explicit_mass_com_inertia_and_principal_axes": explicit_mass_properties,
        "body_mass_kg": float(body_mass.GetMassAttr().Get()),
        "body_center_of_mass_m": [
            float(value) for value in body_mass.GetCenterOfMassAttr().Get()
        ],
        "body_diagonal_inertia_kg_m2": [
            float(value) for value in body_mass.GetDiagonalInertiaAttr().Get()
        ],
        "nut_mass_kg": float(nut_mass.GetMassAttr().Get()),
        "nut_center_of_mass_m": [
            float(value) for value in nut_mass.GetCenterOfMassAttr().Get()
        ],
        "nut_diagonal_inertia_kg_m2": [
            float(value) for value in nut_mass.GetDiagonalInertiaAttr().Get()
        ],
    }
    result["passed"] = bool(
        result["socket_count"] == 61
        and result["key_count"] == 5
        and result["all_socket_fronts_ahead_of_opaque_face"]
        and result["continuous_grip_collision_present"]
        and result["continuous_grip_collision_visibility"] == "invisible"
        and result["explicit_mass_com_inertia_and_principal_axes"]
    )
    return result


def _capture_view(
    *,
    rep,
    ViewportManager,
    camera,
    fixed_imageable,
    loose_imageable,
    view,
    warmup_frames: int,
    rt_subframes: int,
):
    if view["show"] == "loose":
        loose_imageable.MakeVisible()
        fixed_imageable.MakeInvisible()
    else:
        fixed_imageable.MakeVisible()
        loose_imageable.MakeInvisible()
    ViewportManager.set_camera_view(
        camera=camera,
        eye=np.asarray(view["eye_m"], dtype=np.float64),
        target=np.asarray(view["target_m"], dtype=np.float64),
    )
    product = rep.create.render_product(
        camera.GetPrim(),
        RESOLUTION,
        name=f"CheckpointA_{view['name']}",
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
        if rgba.ndim != 3 or rgba.shape[:2] != RESOLUTION[::-1] or rgba.shape[2] < 3:
            raise RuntimeError(f"invalid RGB frame for {view['name']}: {rgba.shape}")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        maximum = np.max(rgb, axis=2)
        diagnostics = {
            "eye_m": list(view["eye_m"]),
            "target_m": list(view["target_m"]),
            "shown_endpoint": view["show"],
            "shape": list(rgb.shape),
            "mean_rgb": [float(value) for value in np.mean(rgb, axis=(0, 1))],
            "std_rgb": [float(value) for value in np.std(rgb, axis=(0, 1))],
            "near_black_pixel_fraction": float(np.mean(maximum <= 2)),
            "nonblack_passed": bool(np.mean(maximum <= 2) < 0.95),
        }
        return rgb, diagnostics
    finally:
        try:
            annotator.detach([product.path])
        finally:
            product.destroy()


def main(argv=None) -> int:
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
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    try:
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.rendering_manager import ViewportManager
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

        context = omni.usd.get_context()
        if context.open_stage(str(asset)) is not True:
            raise RuntimeError("failed to open checkpoint-A asset")
        for _ in range(3):
            app.update()
        stage = context.get_stage()
        fixed_imageable = UsdGeom.Imageable(_prim(stage, FIXED))
        loose_imageable = UsdGeom.Imageable(_prim(stage, LOOSE))
        camera = UsdGeom.Camera.Define(stage, CAMERA)
        camera.CreateFocalLengthAttr(28.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1.0))
        dome = UsdLux.DomeLight.Define(stage, "/World/CheckpointAFill")
        dome.CreateIntensityAttr(1100.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/CheckpointAKey")
        key.CreateIntensityAttr(1800.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.88, 0.78))
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(
            Gf.Vec3f(-35.0, 25.0, 30.0)
        )

        structural = _structural_audit(stage, UsdGeom, UsdPhysics)
        output.mkdir(parents=True, exist_ok=False)
        view_reports = {}
        for view in VIEWS:
            rgb, diagnostics = _capture_view(
                rep=rep,
                ViewportManager=ViewportManager,
                camera=camera,
                fixed_imageable=fixed_imageable,
                loose_imageable=loose_imageable,
                view=view,
                warmup_frames=arguments.warmup_frames,
                rt_subframes=arguments.rt_subframes,
            )
            path = output / f"{view['name']}.png"
            Image.fromarray(rgb, "RGB").save(path)
            diagnostics["file"] = path.name
            view_reports[view["name"]] = diagnostics

        nonblack = all(item["nonblack_passed"] for item in view_reports.values())
        report = {
            "schema_version": SCHEMA_VERSION,
            "asset": str(asset),
            "immutable_output_guard": True,
            "structural_audit": structural,
            "views": view_reports,
            "render_nonblack_passed": nonblack,
            "human_visual_review_complete": False,
            "control_authorized": False,
            "hardware_authorized": False,
            "claims": {
                "public_dimensions_and_topology_verified": structural["passed"],
                "simulation_mass_properties_explicit": structural[
                    "explicit_mass_com_inertia_and_principal_axes"
                ],
                "manufacturer_cad_fidelity": False,
                "hardware_mass_property_fidelity": False,
                "thread_collision_modeled": False,
                "physical_grasp_validated_on_current_asset": False,
            },
            "passed_before_human_review": bool(structural["passed"] and nonblack),
        }
        (output / "checkpoint_a_render_report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        return 0 if report["passed_before_human_review"] else 2
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
