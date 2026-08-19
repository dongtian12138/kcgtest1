import time

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
)

import numpy as np
from omni.usd import get_context
from pxr import Gf, Sdf, UsdGeom

stage = get_context().get_stage()
camera_prim = UsdGeom.Camera.Define(stage, "/World/LiveProbeCam")
xform = UsdGeom.Xformable(camera_prim)
xform.ClearXformOpOrder()
matrix = Gf.Matrix4d(1.0)
matrix.SetTranslateOnly(Gf.Vec3d(1.5, 1.2, 0.9))
xform.AddTransformOp().Set(matrix)
camera_prim.CreateFocalLengthAttr(24.0)
camera_prim.CreateHorizontalApertureAttr(20.955)
camera_prim.CreateVerticalApertureAttr(20.955 * 270.0 / 480.0)
camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 20.0))

try:
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import Camera

    product = rep.create.render_product(
        stage.GetPrimAtPath("/World/LiveProbeCam"),
        (480, 270),
        name="LiveProbeProduct",
    )
    print("PROBE product:", product.path, flush=True)
    cam_obj = Camera(
        prim_path="/World/LiveProbeCam",
        name="live_probe",
        frequency=15,
        resolution=(480, 270),
        render_product_path=product.path,
    )
    print("PROBE cam created:", cam_obj is not None, flush=True)
    cam_obj.initialize(attach_rgb_annotator=True)
    print("PROBE initialized", flush=True)
    for i in range(10):
        rep.orchestrator.step(
            rt_subframes=1, delta_time=0.0, pause_timeline=False
        )
        time.sleep(0.3)
        rgba = cam_obj.get_rgba(device="cpu")
        print(
            f"PROBE frame {i}:",
            None if rgba is None else (rgba.shape, rgba.dtype),
            flush=True,
        )
except Exception as error:
    import traceback
    traceback.print_exc()
    print("PROBE_ERROR", type(error).__name__, error, flush=True)

time.sleep(2)
simulation_app.close()
