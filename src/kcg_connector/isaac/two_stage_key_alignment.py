"""Axial alignment from current visual poses, without scene-authoring inputs."""
import numpy as np
from scipy.spatial.transform import Rotation
from key_direction_memory import _pose, _unit


def axial_target(world_from_body, world_from_socket):
    """Change only twist about the measured Body axis; preserve center/axis."""
    body, socket = _pose(world_from_body), _pose(world_from_socket)
    axis = body[:3, 2]
    current = _unit(body[:3, 1] - axis * (axis @ body[:3, 1]))
    desired = _unit(socket[:3, 1] - axis * (axis @ socket[:3, 1]))
    angle = float(np.arctan2(axis @ np.cross(current, desired), current @ desired))
    target = body.copy()
    target[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix() @ body[:3, :3]
    return target, {
        'signed_rotation_about_body_axis_deg': float(np.degrees(angle)),
        'current_visual_key_world': body[:3, 1].tolist(),
        'measured_socket_keyway_world': socket[:3, 1].tolist(),
        'body_axis_world': axis.tolist(),
        'preserved_body_center_world_m': body[:3, 3].tolist(),
        'inputs': 'CURRENT_KEY_IMAGE_ENCODER_PALM_AND_CURRENT_WRIST_SLOT_IMAGE',
        'scene_initial_angle_is_not_an_input': True,
    }
