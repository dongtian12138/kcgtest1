"""Read native rigid-link poses for postrun audit; never return them to control."""
import numpy as np


def read_native_hand_link_poses(contact_prim, sensor_paths, robot_root):
    """Use the existing validated tensor view, preserving its sensor-path order.

    Intended for TruthAuditRecorder.capture after a completed physics step.
    Existing FK-derived fields must remain separately named and unchanged.
    """
    required = ("handbase_link", "f1Link3", "f2Link2", "f3Link3")
    indices = {}
    for i, path in enumerate(sensor_paths):
        name = str(path).rsplit("/", 1)[-1]
        if str(path).startswith(robot_root.rstrip("/")+"/") and name in required:
            if name in indices:
                raise ValueError(f"ambiguous native robot link: {name}")
            indices[name] = i
    # The contact view need not include the palm if it has no sensor shape.
    if not all(name in indices for name in required[1:]):
        raise ValueError("existing native contact view omits a terminal link")
    if not contact_prim.is_physics_tensor_entity_valid():
        raise RuntimeError("native link pose audit requires an initialized physics tensor view")
    from isaacsim.core.experimental.utils.backend import use_backend
    with use_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
        positions, orientations = contact_prim.get_world_poses()
    positions, orientations = positions.numpy(), orientations.numpy()
    if positions.shape != (len(sensor_paths), 3) or orientations.shape != (len(sensor_paths), 4):
        raise RuntimeError("native rigid poses do not match the validated sensor-path order")
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise RuntimeError("nonfinite native rigid pose audit")
    return {
        "source": "EXISTING_NATIVE_PHYSICS_TENSOR_RIGID_LINK_TRANSFORMS",
        "used_for_online_control": False,
        "poses": {name: {"position_world_m": positions[i].tolist(),
                          "orientation_world_wxyz": orientations[i].tolist(),
                          "sensor_index": i, "prim_path": str(sensor_paths[i])}
                  for name, i in indices.items()},
    }
