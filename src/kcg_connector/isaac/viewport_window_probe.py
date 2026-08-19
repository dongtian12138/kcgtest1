import time

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
)

import omni.kit.app
import omni.ui
from omni.usd import get_context
from pxr import Sdf, UsdGeom

stage = get_context().get_stage()
UsdGeom.Camera.Define(stage, "/World/ProbeCam")

from omni.kit.viewport.utility import create_viewport_window

results = []
for index, flags in enumerate(
    (omni.ui.WINDOW_FLAGS_NO_DOCKING, omni.ui.WINDOW_FLAGS_NONE)
):
    window = create_viewport_window(
        f"Probe{index}",
        width=320,
        height=200,
        position_x=60 + index * 380,
        position_y=60,
        camera_path=Sdf.Path("/World/ProbeCam"),
        flags=flags,
    )
    info = {"index": index, "created": window is not None}
    if window is not None:
        for attr in ("docked", "visible", "position_x", "position_y"):
            try:
                info[attr] = getattr(window, attr)
            except Exception as error:
                info[attr] = f"ERR:{error}"
        try:
            info["undock_ret"] = window.undock()
        except Exception as error:
            info["undock_ret"] = f"ERR:{error}"
        window.visible = True
    results.append(info)
    print(f"PROBE_RESULT {info}", flush=True)

print("PROBE_KEEPALIVE 150s", flush=True)
time.sleep(150)
print("PROBE_DONE", flush=True)
simulation_app.close()
