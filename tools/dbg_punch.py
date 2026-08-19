import numpy as np
from kcg_connector.d38999_cad_registration import (
    fixed_camera_model, proxy_cad_points, render_points, transform_points, project,
)
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix

state = np.array([0.52, -0.21, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
camera = fixed_camera_model(eye=(0.52, -0.21, 0.35), target=(0.52, -0.21, 0.25), resolution=(320, 180))
plug_cad, _ = proxy_cad_points()
xyz = plug_cad.xyz[plug_cad.label == 1]
twp = pose_matrix(state[:6])
world = twp[:3,:3] @ xyz.T + twp[:3,3].reshape(3,1)
uv, pred = project(camera, world.T)
print("uv range:", uv[:,0].min().round(1), uv[:,0].max().round(1), uv[:,1].min().round(1), uv[:,1].max().round(1))
print("depth range:", pred.min().round(4), pred.max().round(4))
in_punch = (uv[:,1] >= 130) & (uv[:,1] < 180) & (uv[:,0] >= 60) & (uv[:,0] < 260)
print("points in punch region:", int(in_punch.sum()), "of", len(xyz))
